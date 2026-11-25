## The decision layer: the per-turn loop that asks LLM seats what their
## particles do next, and runs explicit scripted baselines for scripted seats.
##
## Cadence: one turn every `turnTicks` (108 ticks = 4.5 s of sim time), 10
## turns per round, 40 per episode. At each turn the server builds all LLM
## seats' request bodies and issues them as ONE parallel batch
## (`curl.makeRequests`) — particle worlds is a SIMULTANEOUS-decision game, so
## querying seats one after another would quadruple the wall clock for no gain.
##
## NO FABRICATED MOVES. Every wait is bounded (`attempt1Ms`, `retryMs`,
## `turnBudgetMs`). If an LLM seat cannot produce a valid directive (missing
## credentials/endpoint, auth failure, throttle, timeout, transport error, or
## invalid reply after the single retry), the turn records `engine.policyFailure`
## and returns immediately with NO substitute move so the server terminates the
## episode as an explicit policy failure with no participant scores.

import
  std/[json, math, monotimes, os, strutils, times],
  curly,
  sim, control, directives, baselines, llm, prompts, training_capture

type
  SeatPolicy* = object
    ## What one seat registered as.
    isLlm*: bool
    prompt*: string
    model*: string
    baseline*: Baseline
    label*: string
    registered*: bool

  PolicyFailure* = object
    failed*: bool
    seat*: int
    cause*: string
    detail*: string

  DecisionEngine* = object
    capture*: RuntimeCapture
    selection*: PromptSelection
    client*: LlmClient
    ctl*: ControlState
    seats*: seq[SeatPolicy]
    directives*: seq[SquadDirective]
    haveDirective*: seq[bool]
    lastBatchStart*: MonoTime
    batchStarted*: bool
    policyFailure*: PolicyFailure
    records*: seq[string]      ## chat records queued for the replay writer

proc initDecisionEngine*(sim: SimServer, selection: PromptSelection): DecisionEngine =
  result.selection = selection
  result.capture = newRuntimeCapture("particle-worlds", sim.config.seed)
  result.client = newLlmClient(sim.config)
  result.ctl = initControlState(sim)
  result.seats = newSeq[SeatPolicy](sim.seatCount())
  result.directives = newSeq[SquadDirective](sim.seatCount())
  result.haveDirective = newSeq[bool](sim.seatCount())
  for i in 0 ..< result.seats.len:
    result.seats[i].baseline = blDrifter
    result.seats[i].label = "drifter"

proc policyKind*(engine: DecisionEngine, seat: int): string =
  if seat >= 0 and seat < engine.seats.len and engine.seats[seat].isLlm:
    "llm"
  else:
    "scripted"

# ---------------------------------------------------------------------------
#  The per-seat view
# ---------------------------------------------------------------------------

proc secretJson(sim: SimServer, seat: int): JsonNode =
  ## The ONE mode-conditional block, and the whole entitlement matrix. A seat
  ## that is not entitled sees `null` — never an ABSENT key — so a model never
  ## has to distinguish "hidden" from "malformed".
  ## tests/test_observation.nim asserts this from both sides.
  result = %*{"goal": newJNull(), "goal_colour": newJNull(),
              "key": newJNull()}
  let role = sim.roleIndex[min(max(0, seat), 3)]
  case sim.mode
  of modeSpread, modeTag:
    discard                        ## nothing extra: there is no secret.
  of modeDeceive:
    if role != 0:                  ## the three good agents are told the goal
      result["goal"] = %sim.goalLandmark
      result["goal_colour"] = %MarkColourNames[max(0, sim.goalColour())]
    else:
      result["goal_is_one_of"] = %[0, 1, 2, 3]
  of modeCrypto:
    if role == 0:                  ## Alice sees the goal, its colour AND the key
      result["goal"] = %sim.goalLandmark
      result["goal_colour"] = %MarkColourNames[max(0, sim.goalColour())]
    if role == 0 or role == 1:     ## Alice and Bob share the key
      var key = newJArray()
      for colour in 0 ..< LandmarkCount:
        key.add(%[%symbolText(sim.keySymbols[colour]),
                  %MarkColourNames[colour]])
      result["key"] = key

proc seatViewJson*(
  engine: DecisionEngine,
  sim: SimServer,
  seat, turnIndex, turnsPerRound: int
): string =
  ## Everything this seat may legitimately know, in map pixels, rounded to
  ## integers.
  ##
  ## Positions are FULLY OBSERVABLE — MPE is a fully observable environment,
  ## and hiding positions would add a search puzzle the idea never asks for and
  ## subtract the one it does ask for (inference from BEHAVIOUR and from
  ## SYMBOLS). What is hidden is the round's secret, the other seats' orders for
  ## the turn being decided, every seat's `note`, every PLAYER_PROMPT, every
  ## real policy name, the seed and the RNG state.
  let
    role = sim.roleIndex[min(max(0, seat), 3)]
    played = sim.gameTicksElapsed() div TargetFps
    total = (if sim.config.maxTicks > 0: sim.config.maxTicks div TargetFps
             else: 0)
    (px, py) = sim.particleCentre(seat)
    (cx, cy) = centreOfField()

  var marks = newJArray()
  for i in 0 ..< sim.landmarks.len:
    marks.add(%*{
      "i": i,
      "pos": [sim.landmarks[i].x, sim.landmarks[i].y],
      "colour": MarkColourNames[sim.landmarks[i].colour],
      "r": sim.config.landmarkRadius
    })

  var agents = newJArray()
  for i in 0 ..< min(4, sim.players.len):
    let (ax, ay) = sim.particleCentre(i)
    agents.add(%*{
      "id": sim.cogAlias(i),
      "role": roleName(sim.mode, sim.roleIndex[i]),
      "anchored": sim.isAnchored(i),
      "pos": [ax, ay],
      "vel": [sim.players[i].velX, sim.players[i].velY],
      "colour": teamText(sim.players[i].team)
    })

  var radio = newJArray()
  for i in 0 ..< min(4, sim.players.len):
    radio.add(%*{
      "id": sim.cogAlias(i),
      "now": symbolText(sim.commSymbol[i]),
      "last": symbolText(sim.commPrev[i])
    })

  ## Public BEHAVIOUR: which mark each mobile agent is nearest and how long it
  ## has stayed there. This is the legitimate signal an eavesdropper tails and
  ## the listener must confound, so every seat gets it in every mode.
  var beliefs = newJArray()
  for i in 0 ..< min(4, sim.players.len):
    if sim.isAnchored(i):
      continue
    beliefs.add(%*{
      "id": sim.cogAlias(i),
      "nearest_mark": sim.nearestMark[i],
      "settled_ticks": sim.settledTicks[i]
    })

  var banked = newJArray()
  for entry in sim.roundLog:
    banked.add(%(entry.permille[min(max(0, seat), 3)].float / 1000.0))
  ## `tag` scores from the contact counters at round end and never writes
  ## `roundAccum` (scoring.nim:132), so reading roundAccum told every seat in
  ## every tag round that its round score so far was 0.000 while the spectator
  ## frame showed the real number. The seat view computes it the way
  ## broadcast.nim:1052 does, from the same live term.
  let
    elapsed = max(1, sim.gameTicksElapsed())
    scoreSeat = min(max(0, seat), 3)
    soFar =
      if sim.mode == modeTag: sim.tagRoundPermille(scoreSeat, elapsed)
      else: clamp(int(sim.roundAccum[scoreSeat] div elapsed), 0, 1000)

  var node = %*{
    "round": sim.roundIndex + 1,
    "of": max(1, sim.config.maxGames),
    "mode": $sim.mode,
    "turn": turnIndex,
    "turns": turnsPerRound,
    "clock": {"played_s": played, "left_s": max(0, total - played)},
    "field": {"w": MapWidth, "h": MapHeight, "centre": [cx, cy]},
    "you": {
      "id": sim.cogAlias(seat),
      "role": roleName(sim.mode, role),
      "anchored": sim.isAnchored(seat),
      "pos": [px, py],
      "vel": [sim.players[seat].velX, sim.players[seat].velY],
      "speed_px_s": (abs(sim.players[seat].velX) + abs(sim.players[seat].velY)) *
        TargetFps div max(1, sim.config.motionScale),
      "accel_px_s2": sim.particleAccel(seat) * TargetFps * TargetFps div
        max(1, sim.config.motionScale),
      "max_px_s": sim.particleMaxSpeed(seat) * TargetFps div
        max(1, sim.config.motionScale)
    },
    "marks": marks,
    "agents": agents,
    "radio": radio,
    "secret": secretJson(sim, seat),
    "beliefs": beliefs,
    "score": {
      "this_round_so_far": soFar.float / 1000.0,
      "rounds_banked": banked,
      "episode_so_far": sim.episodePermille(seat).float / 1000.0
    }
  }
  case sim.mode
  of modeSpread:
    node["cover_pct"] = %(sim.coverPermille() div 10)
    node["bumps"] = %sim.bumps[min(max(0, seat), 3)]
  of modeTag:
    var contact = newJArray()
    for i in 0 ..< min(4, sim.players.len):
      contact.add(%sim.tagContact[i])
    node["contact"] = contact
    node["tag_ticks"] = %sim.tagTicks
  else:
    discard
  if seat < engine.haveDirective.len and engine.haveDirective[seat]:
    node["your_last_directive"] = %engine.directives[seat].note
  else:
    node["your_last_directive"] = newJNull()
  $node

# ---------------------------------------------------------------------------
#  Records
# ---------------------------------------------------------------------------

proc registerRecord*(
  seat: int, alias, colour, policy, kind, baseline: string
): string =
  ## The REDACTED registration record. The seat's prompt is never written:
  ## only the policy label, the kind, and which baseline a scripted seat
  ## picked.
  $(%*{
    "k": "register",
    "seat": seat,
    "alias": alias,
    "colour": colour,
    "policy": policy.truncateRunes(MaxPolicyLabelRunes),
    "kind": kind,
    "baseline": baseline
  })

proc roundcardRecord*(sim: SimServer): string =
  ## The `roundcard` control record: the round's mode, its four public roles,
  ## its goal and colour, its key and its marks. A CONVENIENCE for
  ## tools/replay_summary.py — the ONLY reader in the tree. Playback DROPS it
  ## (`replays.nim`'s chat branch keeps `directive` and the `stop` record and
  ## nothing else), so nothing cross-checks it against a re-derivation: the
  ## viewer re-derives all of these values from the seeded draw, and every one
  ## of them is in `gameHash`, so a divergence surfaces as a hash mismatch at
  ## the tick it happens rather than as a record comparison.
  var
    roles = newJArray()
    marks = newJArray()
  for seat in 0 ..< 4:
    roles.add(%roleName(sim.mode, sim.roleIndex[seat]))
  for mark in sim.landmarks:
    marks.add(%[%mark.x, %mark.y, %MarkColourNames[mark.colour]])
  var key: JsonNode = newJNull()
  if sim.mode == modeCrypto:
    key = newJArray()
    for colour in 0 ..< LandmarkCount:
      key.add(%[%symbolText(sim.keySymbols[colour]),
                %MarkColourNames[colour]])
  $(%*{
    "k": "roundcard",
    "round": sim.roundIndex + 1,
    "mode": $sim.mode,
    "roles": roles,
    "goal": sim.goalLandmark,
    "goal_colour": (
      if sim.goalColour() >= 0: %MarkColourNames[sim.goalColour()]
      else: newJNull()),
    "key": key,
    "marks": marks
  })

proc resultRecord*(sim: SimServer): string =
  ## The `result` control record — the episode's whole results document,
  ## written once into the replay chat stream at episode end (design §Record
  ## vocabulary, docs/PROTOCOL.md §The replay). It is what makes the replay
  ## SELF-SUFFICIENT: without it the outcome exists only at
  ## COGAME_RESULTS_URI, and `replay_summary.py`'s `results` reads `{}` for a
  ## spectator holding the bytes. The document is already valid JSON, so it is
  ## embedded verbatim rather than re-parsed: nothing on the path to the
  ## artifact writes may raise.
  "{\"k\":\"result\",\"results\":" & sim.playerResultsJson() & "}"

# ---------------------------------------------------------------------------
#  The turn
# ---------------------------------------------------------------------------

proc scriptedFor*(
  engine: DecisionEngine, sim: SimServer, seat: int, kind: Baseline
): SquadDirective =
  scriptedDirective(engine.ctl, sim, kind, sim.commandedCogs(seat))

proc drifterFor*(
  engine: DecisionEngine, sim: SimServer, cogs: seq[int]
): SquadDirective =
  ## The published `drifter` directive for an explicit scripted particle set.
  scriptedDirective(engine.ctl, sim, blDrifter, cogs)

proc turn*(
  engine: var DecisionEngine,
  sim: SimServer,
  turnIndex, turnsPerRound: int,
  elapsedSeconds: int
): seq[string] =
  ## Runs ONE decision turn and installs each seat's directive. Explicit
  ## scripted seats run their chosen baseline. LLM seats must return a strict,
  ## valid directive within the bounded budget; any failure records
  ## `engine.policyFailure` with NO substitute move so the server terminates
  ## the episode as an explicit policy failure without publishing scores.
  discard elapsedSeconds
  let budget = initDuration(milliseconds = max(1, sim.config.turnBudgetMs))
  var turnStart = getMonoTime()
  engine.client.throttled = false
  engine.policyFailure = PolicyFailure()

  if engine.capture != nil:
    var observations: seq[JsonNode]
    for seat in 0 ..< engine.seats.len:
      observations.add(parseJson(engine.seatViewJson(sim, seat, turnIndex, turnsPerRound)))
    engine.capture.beginTurn(observations)

  # --- which seats need a call? --------------------------------------------
  var open: seq[int]
  for seat in 0 ..< engine.seats.len:
    if engine.seats[seat].isLlm:
      if engine.client.disabled:
        let detail = "LLM endpoint/credentials unavailable for model seat " & $seat
        engine.capture.failedDecision(seat, sim.tickCount)
        engine.policyFailure = PolicyFailure(
          failed: true,
          seat: seat,
          cause: "no_credentials",
          detail: detail
        )
        echo "particle-worlds llm: seat ", seat,
          " terminal policy failure (no_credentials): ", detail
        return
      open.add(seat)
    else:
      var directive = engine.scriptedFor(
        sim, seat, engine.seats[seat].baseline)
      directive.source = dsScripted
      engine.directives[seat] = directive
      engine.haveDirective[seat] = true

  # --- the rate floor -------------------------------------------------------
  if open.len > 0 and engine.batchStarted and sim.config.turnSpacingMs > 0:
    let since = (getMonoTime() - engine.lastBatchStart).inMilliseconds.int
    if since < sim.config.turnSpacingMs:
      sleep(min(sim.config.turnSpacingMs, sim.config.turnSpacingMs - since))
  if open.len > 0:
    engine.lastBatchStart = getMonoTime()
    engine.batchStarted = true
    turnStart = engine.lastBatchStart

  # --- up to two PARALLEL batches ------------------------------------------
  var
    lastCause = newSeq[string](engine.seats.len)
    lastDetail = newSeq[string](engine.seats.len)
  var attempt = 0
  while open.len > 0 and attempt < 2:
    if engine.client.disabled:
      break
    if getMonoTime() - turnStart >= budget:
      for seat in open:
        lastCause[seat] = "timeout"
        lastDetail[seat] =
          "per-turn budget exhausted before attempt " & $(attempt + 1)
      break
    let deadlineMs =
      if attempt == 0: sim.config.attempt1Ms else: sim.config.retryMs
    var batch: RequestBatch
    for seat in open:
      var user = engine.seatViewJson(sim, seat, turnIndex, turnsPerRound)
      if attempt > 0:
        user.add("\n\nYour previous reply was not usable. Reply with ONLY " &
          "the JSON object described above, starting with '{', with exactly " &
          "one \"cogs\" entry, for yourself.")
      let request = engine.client.requestFor(
        systemPrompt(engine.selection),
        userMessage(engine.seats[seat].prompt, user),
        seat,
        engine.seats[seat].model)
      batch.post(request.url, request.headers, request.body, $seat)
    let started = getMonoTime()
    let responses = engine.client.curl.makeRequests(
      batch, max(1, deadlineMs div 1000))
    let latency = (getMonoTime() - started).inMilliseconds.int
    var stillOpen: seq[int]
    for position, seat in open:
      var cause = "parse_error"
      var evidence = modelAttempt($sim.tickCount & "-" & $seat & "-" & $(attempt + 1),
        engine.seats[seat].label,
        engine.client.requestedModel(engine.seats[seat].model),
        batch[position].url,
        batch[position].body, responses[position].response, responses[position].error, latency)
      try:
        let text = engine.client.textOf(
          responses[position].response, responses[position].error,
          batch[position].url)
        evidence.response = %text
        let commanded = sim.commandedCogs(seat)
        var ids: seq[string]
        for cogIndex in commanded:
          ids.add(sim.cogAlias(cogIndex))
        let (fx, fy) = centreOfField()
        var directive = parseSquadDirective(
          extractJsonObject(text), ids, commanded,
          fx, fy, MapWidth - 1, MapHeight - 1)
        directive.source = dsLlm
        directive.latencyMs = latency
        evidence.parsedAction = directive.actionJson()
        evidence.accepted = true
        engine.directives[seat] = directive
        engine.haveDirective[seat] = true
      except CatchableError as error:
        evidence.rejectionReason = some(error.msg)
        if responses[position].error.len > 0:
          cause = (if "timeout" in responses[position].error.toLowerAscii():
                     "timeout" else: "transport_error")
        elif error.msg.startsWith("llm throttled"):
          cause = "throttled"
        elif error.msg.startsWith("llm auth failed"):
          cause = "auth_error"
        lastCause[seat] = cause
        lastDetail[seat] = error.msg
        echo "particle-worlds llm: seat ", seat, " attempt ", attempt + 1,
          " failed: ", error.msg
        stillOpen.add(seat)
      engine.capture.recordAttempt(seat, evidence)
    open = stillOpen
    inc attempt
    if engine.client.throttled and open.len > 0:
      echo "particle-worlds llm: provider throttled with no other candidate; ",
        open.len, " seat(s) failed on turn ", turnIndex
      break

  # --- any seat still open is a terminal policy failure (no substitute move)
  if open.len > 0:
    for seat in open:
      engine.capture.failedDecision(seat, sim.tickCount)
    let failedSeat = open[0]
    let cause =
      if lastCause[failedSeat].len > 0: lastCause[failedSeat]
      elif engine.client.disabled or engine.client.transport == ltNone: "no_credentials"
      elif engine.client.throttled: "throttled"
      else: "parse_error"
    let detail =
      if lastDetail[failedSeat].len > 0:
        lastDetail[failedSeat].truncateRunes(MaxFallbackDetailRunes)
      else: "model failed to produce a valid directive"
    engine.policyFailure = PolicyFailure(
      failed: true,
      seat: failedSeat,
      cause: cause,
      detail: detail
    )
    echo "particle-worlds llm: seat ", failedSeat,
      " terminal policy failure (", cause, ") on turn ", turnIndex, ": ", detail
    return

  for seat in 0 ..< engine.seats.len:
    let origin = if engine.directives[seat].source == dsLlm: aoModel
      else: aoTeacher
    engine.capture.applied(seat, sim.tickCount, engine.directives[seat].actionJson(), origin,
      if origin == aoTeacher: "scripted-" & $engine.seats[seat].baseline
      else: engine.seats[seat].label)
