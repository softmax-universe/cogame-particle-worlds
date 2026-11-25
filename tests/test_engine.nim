## The turn loop, against a REAL fake provider.
##
## Proves:
## * All LLM seats' calls go out as ONE parallel batch per turn.
## * Explicit scripted seats make zero LLM calls.
## * Invalid output retries once, then ends in terminal policyFailure with NO
##   substitute move (neither hold nor drifter).
## * Throttling (429), timeout, and missing credentials all end in terminal
##   policyFailure with NO substitute move.
## * Rate floor and per-turn budget bounds are honoured.

import std/[atomics, json, locks, monotimes, os, strutils, times, unicode,
            unittest]
import bitworld/spriteprotocol
import mummy, mummy/routers
import ../src/mpe/[sim, control, directives, baselines, decide, llm, prompts]
import fixture

type
  Window = object
    startMs, endMs: int

var
  windowLock: Lock
  windows: seq[Window]
  serverStarted: MonoTime
  holdMs: Atomic[int]           ## how long the fake sleeps before answering
  statusCode: Atomic[int]       ## what the fake answers with
  forceBadBody: Atomic[bool]
  replyBody: string
  bodyLock: Lock

windowLock.initLock()
bodyLock.initLock()

proc nowMs(): int =
  (getMonoTime() - serverStarted).inMilliseconds.int

proc bedrockReply(text: string): string =
  $(%*{"content": [{"type": "text", "text": text}]})

proc directiveReply(alias: string): string =
  bedrockReply($(%*{
    "note": "fake provider",
    "cogs": [{"id": alias, "intent": "cover", "target": [400, 300],
              "face": newJNull(), "symbol": "F"}]
  }))

proc aliasFromRequestBody(reqBody: string): string =
  try:
    let node = parseJson(reqBody)
    let user = node["messages"][0]["content"].getStr()
    let idx = user.find("\"you\":")
    if idx >= 0:
      let idIdx = user.find("\"id\":\"", idx)
      if idIdx >= 0:
        let startPos = idIdx + 6
        let endPos = user.find("\"", startPos)
        if endPos > startPos:
          return user[startPos ..< endPos]
  except CatchableError:
    discard
  "RED-alpha"

proc fakeHandler(request: Request) {.gcsafe.} =
  let began = nowMs()
  let hold = holdMs.load()
  if hold > 0:
    sleep(hold)
  let code = statusCode.load()
  var body: string
  {.cast(gcsafe).}:
    if forceBadBody.load():
      withLock bodyLock:
        body = replyBody
    else:
      body = directiveReply(aliasFromRequestBody(request.body))
    withLock windowLock:
      windows.add(Window(startMs: began, endMs: nowMs()))
  var headers: HttpHeaders
  headers["content-type"] = "application/json"
  request.respond(code, headers, body)

var router: Router
router.get("/**", fakeHandler)
router.post("/**", fakeHandler)
let fake = newServer(router)
var servePort = 0

const FakePort = 8791

var fakeThread: Thread[void]
proc serveFake() {.thread.} =
  fake.serve(Port(FakePort))

proc bootFake() =
  serverStarted = getMonoTime()
  holdMs.store(0)
  statusCode.store(200)
  forceBadBody.store(false)
  withLock bodyLock:
    replyBody = directiveReply("RED-alpha")
  createThread(fakeThread, serveFake)
  sleep(400)                     ## let mummy bind before the first request
  servePort = FakePort
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:" & $FakePort)
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
  delEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME")
  delEnv("AWS_BEARER_TOKEN_BEDROCK")
  delEnv("ANTHROPIC_API_KEY")
  delEnv("ANTHROPIC_API_KEY_URI")

proc resetWindows() =
  withLock windowLock:
    windows.setLen(0)

proc recordedWindows(): seq[Window] =
  withLock windowLock:
    result = windows

proc llmEngine(sim: SimServer): DecisionEngine =
  result = initDecisionEngine(sim, PromptSelection())
  for seat in 0 ..< result.seats.len:
    result.seats[seat].isLlm = true
    result.seats[seat].prompt = "cover a mark"
    result.seats[seat].registered = true
    result.seats[seat].label = "fake"

suite "the turn loop":

  bootFake()

  test "the client picks up the native sidecar endpoint":
    var sim = seatedSim(fixtureConfig())
    let client = newLlmClient(sim.config)
    check client.transport == ltSidecar
    check not client.disabled

  test "all four seats' calls go out in ONE parallel batch":
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = llmEngine(sim)
    const Hold = 400
    holdMs.store(Hold)
    resetWindows()
    let began = getMonoTime()
    let records = engine.turn(sim, 0, 10, 0)
    let elapsed = (getMonoTime() - began).inMilliseconds.int
    holdMs.store(0)
    let seen = recordedWindows()
    check seen.len == FixtureSeats
    check elapsed < FixtureSeats * Hold
    var overlaps = 0
    for i in 0 ..< seen.len:
      for j in i + 1 ..< seen.len:
        if seen[i].startMs <= seen[j].endMs and seen[j].startMs <= seen[i].endMs:
          inc overlaps
    check overlaps > 0
    check not engine.policyFailure.failed
    for seat in 0 ..< 4:
      check engine.haveDirective[seat]
      check engine.directives[seat].source == dsLlm
      check engine.directives[seat].orders.len == 1
      check engine.directives[seat].orders[0].intent == intCover
    check records.len == 0

  test "mixed lineup calls LLM only for model seat, baseline seats stay scripted":
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = initDecisionEngine(sim, PromptSelection())
    engine.seats[0].isLlm = true
    engine.seats[0].prompt = "cover a mark"
    engine.seats[0].registered = true
    engine.seats[0].label = "model"
    for seat in 1 ..< 4:
      engine.seats[seat].isLlm = false
      engine.seats[seat].baseline = blDrifter
      engine.seats[seat].registered = true
      engine.seats[seat].label = "drifter"
    resetWindows()
    discard engine.turn(sim, 0, 10, 0)
    check not engine.policyFailure.failed
    check recordedWindows().len == 1
    check engine.directives[0].source == dsLlm
    for seat in 1 ..< 4:
      check engine.directives[seat].source == dsScripted

  test "an unusable reply retries once, then ends in terminal policyFailure with no substitute":
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = llmEngine(sim)
    withLock bodyLock:
      replyBody = bedrockReply("I would rather not answer in JSON.")
    forceBadBody.store(true)
    resetWindows()
    discard engine.turn(sim, 1, 10, 0)
    forceBadBody.store(false)
    ## Two batches of four: attempt 1 and exactly ONE retry.
    check recordedWindows().len == 8
    check engine.policyFailure.failed
    check engine.policyFailure.seat == 0
    check engine.policyFailure.cause == "parse_error"
    for seat in 0 ..< 4:
      check not engine.haveDirective[seat]

  test "a provider error (400 invalid model) ends in terminal policyFailure with provider_error":
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = llmEngine(sim)
    statusCode.store(400)
    resetWindows()
    discard engine.turn(sim, 2, 10, 0)
    statusCode.store(200)
    check engine.policyFailure.failed
    check engine.policyFailure.cause == "provider_error"
    for seat in 0 ..< 4:
      check not engine.haveDirective[seat]

  test "a throttle (429) skips the retry and ends in terminal policyFailure":
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = llmEngine(sim)
    statusCode.store(429)
    resetWindows()
    discard engine.turn(sim, 2, 10, 0)
    statusCode.store(200)
    check recordedWindows().len == 4
    check engine.policyFailure.failed
    check engine.policyFailure.cause == "throttled"
    for seat in 0 ..< 4:
      check not engine.haveDirective[seat]

  test "the per-turn budget is enforced against a hung provider and ends in terminal policyFailure":
    var config = fixtureConfig(@[modeSpread])
    config.attempt1Ms = 1000
    config.retryMs = 1000
    config.turnBudgetMs = 2000
    var sim = seatedSim(config)
    var engine = llmEngine(sim)
    holdMs.store(4000)
    let began = getMonoTime()
    discard engine.turn(sim, 3, 10, 0)
    let elapsed = (getMonoTime() - began).inMilliseconds.int
    holdMs.store(0)
    check elapsed < 3500
    check engine.policyFailure.failed
    check engine.policyFailure.cause == "timeout"
    for seat in 0 ..< 4:
      check not engine.haveDirective[seat]

  test "sim_config rejects deadlines that are not whole seconds":
    var config = fixtureConfig()
    config.attempt1Ms = 4500
    expect MpeError:
      config.update("{}")
    config = fixtureConfig()
    config.retryMs = 2500
    expect MpeError:
      config.update("{}")

  test "sim_config rejects attempt1Ms + retryMs past turnBudgetMs":
    var config = fixtureConfig()
    config.attempt1Ms = 8000
    config.retryMs = 5000
    config.turnBudgetMs = 10_000
    expect MpeError:
      config.update("{}")

  test "the rate floor holds four seats under 30 requests a minute":
    let config = variantConfig(@[modeSpread])
    check config.turnSpacingMs == DefaultParticleTurnSpacingMs
    let perMinute = FixtureSeats * 60_000 div config.turnSpacingMs
    check perMinute <= 30
    var spaced = fixtureConfig(@[modeSpread])
    spaced.turnSpacingMs = 600
    var sim = seatedSim(spaced)
    var engine = llmEngine(sim)
    let began = getMonoTime()
    discard engine.turn(sim, 0, 10, 0)
    discard engine.turn(sim, 1, 10, 0)
    let elapsed = (getMonoTime() - began).inMilliseconds.int
    check elapsed >= spaced.turnSpacingMs

  test "the rate floor never eats the single retry":
    var config = fixtureConfig(@[modeSpread])
    config.turnSpacingMs = 1200
    config.attempt1Ms = 1000
    config.retryMs = 1000
    config.turnBudgetMs = 2000
    var sim = seatedSim(config)
    var engine = llmEngine(sim)
    discard engine.turn(sim, 0, 10, 0)
    resetWindows()
    holdMs.store(4000)
    let spacedBegan = getMonoTime()
    discard engine.turn(sim, 1, 10, 0)
    let spacedElapsed = (getMonoTime() - spacedBegan).inMilliseconds.int
    holdMs.store(0)
    check spacedElapsed >= config.turnSpacingMs
    check spacedElapsed < config.turnSpacingMs + config.turnBudgetMs + 1000
    check engine.policyFailure.failed
    check engine.policyFailure.cause == "timeout"
    check "per-turn budget exhausted" notin engine.policyFailure.detail
    sleep(1500)
    resetWindows()

  test "a seat with NO credentials ends immediately in terminal policyFailure":
    delEnv("COWORLD_LLM_ENDPOINT")
    var sim = seatedSim(fixtureConfig(@[modeSpread]))
    var engine = llmEngine(sim)
    check engine.client.transport == ltNone
    check engine.client.disabled
    discard engine.turn(sim, 0, 10, 0)
    check engine.policyFailure.failed
    check engine.policyFailure.seat == 0
    check engine.policyFailure.cause == "no_credentials"
    for seat in 0 ..< 4:
      check not engine.haveDirective[seat]
    putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:" & $FakePort)
