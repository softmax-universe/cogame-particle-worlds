## The directive schema: what a commander (LLM or scripted) may say, and strict
## parsing of model replies with no fabricated actions or silent coercions.
##
## Both policy kinds emit the SAME object, so the two are strictly comparable
## and one validator covers both — that is what makes the bounded-orders test
## in tests/test_control.nim meaningful.
##
## RUNE DISCIPLINE. Every cap in this file is measured in RUNES (Unicode
## codepoints) and every truncation lands on a rune boundary (`runeLen` /
## `runeSubStr`). Slicing a string by BYTE index anywhere on the path to the
## replay is forbidden: a byte-truncated multi-byte character renders fine in a
## browser and then fails a strict UTF-8 parser, which is exactly the class of
## bug that makes a replay unreadable to everything except the one viewer that
## happened to be lenient.

import
  std/[json, strutils, unicode],
  sim_types

type
  Intent* = enum
    ## What a particle is being told to do for the next turn. A closed enum: an
    ## unrecognised intent is repaired to `go`, never dropped, so a particle is
    ## never left unactuated.
    intGo = "go"
    intHold = "hold"
    intCover = "cover"
    intShadow = "shadow"
    intEvade = "evade"
    intOrbit = "orbit"

  CogOrder* = object
    ## One commanded cog's orders for the next turn.
    cogIndex*: int             ## the live cog this order drives.
    id*: string                ## the cog's anonymous alias, e.g. "RED-alpha".
    intent*: Intent
    targetX*, targetY*: int    ## clamped into the map box.
    hasFace*: bool
    faceX*, faceY*: int
    symbol*: int               ## 0 = silence, 1..8 = A..H. THE ONLY thing a
                               ## seat can say to another seat, and it is
                               ## broadcast to the whole field.
    fromReply*: bool           ## a reply entry really named this particle.
                               ## False means the parser filled it in, and the
                               ## caller repairs it from last turn's directive
                               ## (else `drifter`'s) — never from a default.

  DirectiveSource* = enum
    dsLlm = "llm"
    dsScripted = "scripted"
    dsFallback = "fallback"

  SquadDirective* = object
    ## One seat's whole order set for one turn.
    note*: string              ## <= MaxNoteRunes; the commander's own line.
    orders*: seq[CogOrder]
    source*: DirectiveSource
    latencyMs*: int

  DirectiveError* = object of ValueError

proc truncateRunes*(text: string, limit: int): string =
  ## Cuts `text` to at most `limit` RUNES, on a rune boundary. The single
  ## place any recorded string is shortened.
  if limit <= 0:
    return ""
  if text.runeLen <= limit:
    return text
  text.runeSubStr(0, limit)

proc parseSymbol*(text: string): int =
  ## Strict radio symbol parser: "-" for silence (0), or a single uppercase or
  ## lowercase letter in A..H (1..8). Anything else raises DirectiveError.
  let trimmed = text.strip()
  if trimmed == SymbolSilence:
    return 0
  if trimmed.len == 1:
    let ch = toUpperAscii(trimmed[0])
    for i in 0 ..< SymbolAlphabet.len:
      if SymbolAlphabet[i] == ch:
        return i + 1
  raise newException(DirectiveError, "invalid symbol: " & text.truncateRunes(32))

proc symbolIndexOfText*(text: string): int =
  ## The symbol index of one EXACT wire text ("A".."H"), 0 for anything else.
  if text.len != 1:
    return 0
  for i in 0 ..< SymbolAlphabet.len:
    if SymbolAlphabet[i] == text[0]:
      return i + 1
  0

proc symbolTextOf*(index: int): string =
  ## The wire text of one symbol index: 0 is silence ("-"), 1..8 are A..H.
  if index <= 0 or index > SymbolAlphabet.len:
    SymbolSilence
  else:
    $SymbolAlphabet[index - 1]

proc sanitizeNote*(text: string): string =
  ## The commander's own line, as it reaches the replay and the match feed.
  ## Newlines collapse to spaces so one record stays one line.
  text.replace("\n", " ").replace("\r", " ").strip().truncateRunes(MaxNoteRunes)

proc parseIntent*(text: string): Intent =
  ## Strict intent parser: must match one of the six legal intents.
  let key = text.strip().toLowerAscii()
  for intent in Intent:
    if $intent == key:
      return intent
  raise newException(DirectiveError, "invalid intent: " & text.truncateRunes(32))

proc stripSingleCodeFence(trimmed: string): tuple[ok: bool, body: string] =
  ## Unwraps a single outer markdown code fence (``` or ```json) ONLY when the
  ## entire reply is that one fence with no surrounding prose or inner fences.
  if not trimmed.startsWith("```") or not trimmed.endsWith("```") or trimmed.len < 6:
    return (false, trimmed)
  let nl = trimmed.find('\n')
  if nl < 0 or nl >= trimmed.len - 3:
    return (false, trimmed)
  let tag = trimmed[3 ..< nl].strip().toLowerAscii()
  if tag.len > 0 and tag != "json":
    return (false, trimmed)
  let inner = trimmed[nl + 1 ..< trimmed.len - 3].strip()
  if "```" in inner:
    return (false, trimmed)
  (true, inner)

proc extractJsonObject*(text: string): JsonNode =
  ## Strict JSON object parser: the reply must be a single JSON object
  ## (optionally wrapped in a single outer ``` or ```json fence) with no
  ## surrounding prose and no brace scraping.
  let trimmed = text.strip()
  let (_, candidate) = stripSingleCodeFence(trimmed)
  if candidate.len < 2 or candidate[0] != '{' or candidate[^1] != '}':
    var head = trimmed
    if head.runeLen > 160:
      head = head.truncateRunes(160) & "..."
    raise newException(
      DirectiveError, "reply must be a single JSON object: " & head.replace("\n", " "))
  try:
    result = parseJson(candidate)
  except CatchableError as e:
    raise newException(DirectiveError, "invalid JSON in reply: " & e.msg)
  if result.kind != JObject:
    raise newException(DirectiveError, "reply JSON must be an object")

proc readCoord(node: JsonNode): tuple[ok: bool, value: int] =
  ## One target/face coordinate: an integer or finite number. No string coercion.
  if node.isNil:
    return (false, 0)
  case node.kind
  of JInt:
    (true, int(node.getBiggestInt()))
  of JFloat:
    let f = node.getFloat()
    if f != f or f > 1.0e9 or f < -1.0e9: (false, 0)
    else: (true, int(f))
  else:
    (false, 0)

proc readPoint(
  node: JsonNode, maxX, maxY: int
): tuple[ok: bool, x, y: int] =
  ## Strict `[x, y]` pair, clamped into the map box.
  result = (false, 0, 0)
  if node.isNil or node.kind != JArray or node.len != 2:
    return
  let
    rx = readCoord(node[0])
    ry = readCoord(node[1])
  if not rx.ok or not ry.ok:
    return
  result = (
    true,
    clamp(rx.value, 0, max(0, maxX)),
    clamp(ry.value, 0, max(0, maxY))
  )

proc parseSquadDirective*(
  payload: JsonNode,
  commandedIds: seq[string],
  commandedCogs: seq[int],
  defaultX, defaultY, maxX, maxY: int
): SquadDirective =
  ## Strictly validates one parsed reply into a legal directive. Rejects
  ## missing/invalid fields, unknown intents, invalid symbols, wrong cog counts,
  ## or mismatched cog ids with DirectiveError rather than fabricating a move.
  doAssert commandedIds.len == commandedCogs.len
  discard defaultX
  discard defaultY
  if payload.isNil or payload.kind != JObject:
    raise newException(DirectiveError, "directive payload must be a JSON object")
  let noteNode = payload{"note"}
  if not noteNode.isNil and noteNode.kind notin {JString, JNull}:
    raise newException(DirectiveError, "note must be a string or null")
  result.note = if not noteNode.isNil and noteNode.kind == JString:
    sanitizeNote(noteNode.getStr())
  else:
    ""
  result.source = dsLlm

  let cogsNode = payload{"cogs"}
  if cogsNode.isNil or cogsNode.kind != JArray:
    raise newException(DirectiveError, "cogs must be a JSON array")
  if cogsNode.len != commandedIds.len:
    raise newException(
      DirectiveError,
      "cogs must contain exactly " & $commandedIds.len & " entry")

  var
    claimed = newSeq[bool](commandedIds.len)
    orders = newSeq[CogOrder](commandedIds.len)
  for item in cogsNode:
    if item.kind != JObject:
      raise newException(DirectiveError, "each cogs entry must be a JSON object")
    let idNode = item{"id"}
    if idNode.isNil or idNode.kind != JString or idNode.getStr().strip().len == 0:
      raise newException(DirectiveError, "cogs entry missing string id")
    let wanted = idNode.getStr().strip().toLowerAscii()
    var slot = -1
    for i, id in commandedIds:
      if not claimed[i] and id.toLowerAscii() == wanted:
        slot = i
        break
    if slot < 0:
      raise newException(
        DirectiveError, "cogs entry id does not match commanded cog: " & idNode.getStr())
    claimed[slot] = true

    let intentNode = item{"intent"}
    if intentNode.isNil or intentNode.kind != JString:
      raise newException(DirectiveError, "cogs entry missing string intent")
    let intent = parseIntent(intentNode.getStr())

    let target = readPoint(item{"target"}, maxX, maxY)
    if not target.ok:
      raise newException(DirectiveError, "cogs entry missing valid [x, y] target")

    let faceNode = item{"face"}
    var
      hasFace = false
      faceX = 0
      faceY = 0
    if not faceNode.isNil and faceNode.kind != JNull:
      let face = readPoint(faceNode, maxX, maxY)
      if not face.ok:
        raise newException(DirectiveError, "cogs entry face must be null or [x, y]")
      hasFace = true
      faceX = face.x
      faceY = face.y

    let symbolNode = item{"symbol"}
    if symbolNode.isNil or symbolNode.kind != JString:
      raise newException(DirectiveError, "cogs entry missing string symbol")
    let symbol = parseSymbol(symbolNode.getStr())

    orders[slot] = CogOrder(
      cogIndex: commandedCogs[slot],
      id: commandedIds[slot],
      intent: intent,
      targetX: target.x,
      targetY: target.y,
      hasFace: hasFace,
      faceX: faceX,
      faceY: faceY,
      symbol: symbol,
      fromReply: true
    )
  result.orders = orders

proc directiveRecord*(
  directive: SquadDirective,
  roundIndex, turn, seat: int,
  mode, alias, role: string
): JsonNode =
  ## The replay chat record for one turn's directive. Re-applied at playback
  ## into NON-HASHED sim fields only (the feed and the radio strip), so it
  ## drives the broadcast and tools/replay_summary.py and can never affect the
  ## simulation.
  var cogs = newJArray()
  for order in directive.orders:
    var item = %*{
      "id": order.id,
      "intent": $order.intent,
      "target": [order.targetX, order.targetY],
      "symbol": symbolTextOf(order.symbol)
    }
    if order.hasFace:
      item["face"] = %[order.faceX, order.faceY]
    else:
      item["face"] = newJNull()
    cogs.add(item)
  %*{
    "k": "directive",
    "round": roundIndex,
    "mode": mode,
    "turn": turn,
    "seat": seat,
    "alias": alias,
    "role": role,
    "source": $directive.source,
    "latency_ms": directive.latencyMs,
    "note": directive.note,
    "cogs": cogs
  }

proc boundedDirectiveRecord*(
  directive: SquadDirective,
  roundIndex, turn, seat: int,
  mode, alias, role: string
): string =
  ## The serialized directive record, guaranteed <= MaxDirectiveRunes. The
  ## note is the only unbounded-in-practice field, so it is the one that
  ## shrinks; the cut still lands on a rune boundary.
  var trimmed = directive
  result = $trimmed.directiveRecord(roundIndex, turn, seat, mode, alias, role)
  var guard = 0
  while result.runeLen > MaxDirectiveRunes and guard < 12:
    inc guard
    let keep = max(0, trimmed.note.runeLen - max(8, trimmed.note.runeLen div 2))
    trimmed.note = trimmed.note.truncateRunes(keep)
    result = $trimmed.directiveRecord(
      roundIndex, turn, seat, mode, alias, role)
  # The loop always converges: with an empty note the record is a few hundred
  # runes of fixed structure. Never cut the SERIALIZED string — that would emit
  # broken JSON, which is the exact failure the rune rule exists to prevent.

proc actionJson*(directive: SquadDirective): JsonNode =
  ## The macro the controller receives, excluding spectator and transport metadata.
  var cogs = newJArray()
  for order in directive.orders:
    var item = %*{"id": order.id, "intent": $order.intent,
      "target": [order.targetX, order.targetY]}
    item["face"] = if order.hasFace: %[order.faceX, order.faceY] else: newJNull()
    item["symbol"] = %symbolTextOf(order.symbol)
    cogs.add(item)
  %*{"note": directive.note, "cogs": cogs}
