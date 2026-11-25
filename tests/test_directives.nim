## Strict directive parsing and rune-safe note/record capping.
## No silent coercions, no fabricated moves, no legacy alternate shapes.

import std/[json, strutils, unicode, unittest]
import ../src/mpe/[directives, sim_types]

const
  Ids = @["BLUE-alpha"]
  Cogs = @[1]
  Cx = 617
  Cy = 329
  MaxX = 1234
  MaxY = 658

proc parse(text: string): SquadDirective =
  parseSquadDirective(extractJsonObject(text), Ids, Cogs, Cx, Cy, MaxX, MaxY)

suite "directive parsing":

  test "strict JSON object is accepted and prose/fenced replies are rejected":
    let directive = parse(
      """{"note":"go","cogs":[{"id":"BLUE-alpha","intent":"cover","target":[10,20],"face":null,"symbol":"F"}]}""")
    check directive.orders.len == 1
    check directive.orders[0].intent == intCover
    check directive.orders[0].targetX == 10
    check directive.orders[0].targetY == 20
    check not directive.orders[0].hasFace
    check directive.orders[0].symbol == symbolIndexOfText("F")

    for bad in [
        """Sure! Here is my order:
{"note":"go","cogs":[{"id":"BLUE-alpha","intent":"cover","target":[10,20],"symbol":"F"}]}""",
        """```json
{"note":"go","cogs":[{"id":"BLUE-alpha","intent":"cover","target":[10,20],"symbol":"F"}]}
```""",
        """I refuse to answer in JSON."""]:
      expect DirectiveError:
        discard parse(bad)

  test "cogs as an id-keyed object is rejected":
    expect DirectiveError:
      discard parse(
        """{"cogs":{"BLUE-alpha":{"intent":"orbit","target":[1,2],"symbol":"C"}}}""")

  test "legal intents are accepted and unknown/empty intents are rejected":
    check parseIntent("Cover") == intCover
    check parseIntent(" hold ") == intHold
    check parseIntent("SHADOW") == intShadow
    check parseIntent("go") == intGo
    check parseIntent("evade") == intEvade
    check parseIntent("orbit") == intOrbit
    for bad in ["e-vade", "teleport", ""]:
      expect DirectiveError:
        discard parseIntent(bad)

  test "out-of-bounds targets clamp; missing, NaN, object, or string targets are rejected":
    let clamped = parse(
      """{"cogs":[{"id":"BLUE-alpha","intent":"go","target":[99999,-500],"symbol":"-"}]}""")
    check clamped.orders[0].targetX == MaxX
    check clamped.orders[0].targetY == 0
    for bad in [
        """{"cogs":[{"id":"BLUE-alpha","intent":"go","symbol":"-"}]}""",
        """{"cogs":[{"id":"BLUE-alpha","intent":"go","target":[1e30,-1e30],"symbol":"-"}]}""",
        """{"cogs":[{"id":"BLUE-alpha","intent":"go","target":{"x":"42","y":"7.9"},"symbol":"-"}]}""",
        """{"cogs":[{"id":"BLUE-alpha","intent":"go","target":["42","7"],"symbol":"-"}]}"""]:
      expect DirectiveError:
        discard parse(bad)

  test "wrong cog count or mismatched cog id is rejected":
    expect DirectiveError:
      discard parse("""{"cogs":[
        {"id":"BLUE-alpha","intent":"hold","target":[1,1],"symbol":"A"},
        {"id":"GREEN-alpha","intent":"go","target":[2,2],"symbol":"-"}]}""")
    expect DirectiveError:
      discard parse("""{"note":"nothing to say","cogs":[]}""")
    expect DirectiveError:
      discard parse("""{"note":"nothing to say"}""")
    expect DirectiveError:
      discard parse(
        """{"cogs":[{"id":"GREEN-alpha","intent":"evade","target":[5,5],"symbol":"-"}]}""")

  test "symbols accept '-' or A..H and reject everything else":
    check parseSymbol("f") == symbolIndexOfText("F")
    check parseSymbol("-") == 0
    check symbolTextOf(0) == SymbolSilence
    for i in 1 .. SymbolAlphabet.len:
      check parseSymbol(symbolTextOf(i)) == i
    for bad in ["FF", "Z", "", "   ", "\u2014", "\u{1F680}"]:
      expect DirectiveError:
        discard parseSymbol(bad)

  test "a 300-character note is truncated on a RUNE boundary":
    var long = ""
    for i in 0 ..< 300:
      long.add('x')
    let directive = parse(
      """{"note":"""" & long & """","cogs":[{"id":"BLUE-alpha","intent":"go","target":[10,20],"symbol":"-"}]}""")
    check directive.note.runeLen == MaxNoteRunes
    check directive.note.validateUtf8() == -1

  test "a 4-byte emoji sitting ON the cap survives whole":
    var head = ""
    for i in 0 ..< MaxNoteRunes - 1:
      head.add('a')
    let note = head & "\u{1F680}" & "TAIL"
    check note.runeLen == MaxNoteRunes + 4
    let directive = parse(
      """{"note":"""" & note & """","cogs":[
        {"id":"BLUE-alpha","intent":"cover","target":[3,4],"symbol":"G"}]}""")
    check directive.note.runeLen == MaxNoteRunes
    check directive.note.validateUtf8() == -1
    check directive.note.endsWith("\u{1F680}")
    let record = directive.boundedDirectiveRecord(
      1, 0, 1, "crypto", "BLUE-alpha", "listener")
    check record.runeLen <= MaxDirectiveRunes
    check record.validateUtf8() == -1
    let reparsed = parseJson(record)
    check reparsed["note"].getStr().validateUtf8() == -1
    check reparsed["cogs"][0]["symbol"].getStr() == "G"
    check reparsed["k"].getStr() == "directive"
    check reparsed["role"].getStr() == "listener"

  test "the bounded record shrinks the NOTE, never the serialized string":
    var directive = SquadDirective(source: dsLlm)
    var note = ""
    for i in 0 ..< MaxNoteRunes:
      note.add("\u{1F680}")            ## 160 four-byte runes: 640 bytes
    directive.note = note
    directive.orders = @[CogOrder(
      cogIndex: 1, id: "BLUE-alpha", intent: intCover,
      targetX: 100, targetY: 200, symbol: 6)]
    let record = directive.boundedDirectiveRecord(
      4, 9, 1, "crypto", "BLUE-alpha", "listener")
    check record.runeLen <= MaxDirectiveRunes
    check record.validateUtf8() == -1
    let reparsed = parseJson(record)      ## still legal JSON, never cut
    check reparsed["cogs"][0]["intent"].getStr() == "cover"

  test "sanitizeNote collapses newlines and honours the cap":
    check sanitizeNote("a\nb\r\nc") == "a b  c"
    var long = ""
    for i in 0 ..< 500:
      long.add("\u00e9")                  ## 2-byte runes
    let note = sanitizeNote(long)
    check note.runeLen == MaxNoteRunes
    check note.validateUtf8() == -1
