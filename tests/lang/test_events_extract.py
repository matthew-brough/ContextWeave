from collections.abc import Callable

import pytest

from contextweave.lang.extract_edges import extract_edges
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import ParsedFile

type Parse = Callable[[str, str], ParsedFile]

HANDLERS = {
    "a.lua": 'RegisterNetEvent("e:x", function() end)\nMySQL.query("e:y", {}, function() end)\n'
    'AddEventHandler("e:z" .. n, function() end)\n',
    "a.cs": 'class A { A() { EventHandlers["e:x"] += new Action(() => { });'
    ' db.query("e:y", args, () => { }); } }\n',
    "a.js": 'onNet("e:x", () => {});\nsocket.on("e:y", () => {});\n'
    'on("e:z" + n, () => {});\nfoo("e:w", () => {});\n',
    "a.ts": 'onNet("e:x", (n: number): void => {});\nsocket.on("e:y", () => {});\n',
    "a.tsx": 'on("e:x", (): void => {});\nemitter.on("e:y", () => {});\n',
}


@pytest.mark.parametrize("name", sorted(HANDLERS))
def test_handlers_carry_registrar_and_skip_lookalikes(name: str, parse_text: Parse) -> None:
    symbols = [s for s in extract_symbols(parse_text(name, HANDLERS[name])).symbols if s.registrar]
    assert [s.name for s in symbols] == ["e:x"]


def test_registrar_text_per_language(parse_text: Parse) -> None:
    lua = extract_symbols(parse_text("a.lua", HANDLERS["a.lua"])).symbols
    csharp = extract_symbols(parse_text("a.cs", HANDLERS["a.cs"])).symbols
    assert [s.registrar for s in lua if s.name == "e:x"] == ["RegisterNetEvent"]
    assert [s.registrar for s in csharp if s.name == "e:x"] == ["EventHandlers"]


def test_javascript_wrappers_outside_the_known_list_are_not_handlers(parse_text: Parse) -> None:
    symbols = extract_symbols(parse_text("a.js", HANDLERS["a.js"])).symbols
    assert "e:w" not in {s.name for s in symbols}


EMITS = {
    "a.lua": 'TriggerServerEvent("e:a", 1)\nTriggerClientEvent(src, "e:b")\n'
    'TriggerEvent("e:" .. n)\nTriggerClientEvent("e:" .. n, "late")\nobj.TriggerEvent("e:o")\n',
    "a.cs": 'class A { void F() { TriggerServerEvent("e:a", 1); TriggerClientEvent(p, "e:b");'
    ' TriggerEvent("e:" + n); TriggerClientEvent("e:" + n, "late"); BaseScript.TriggerEvent("e:s");'
    ' obj.TriggerEvent("e:o"); } }\n',
    "a.js": 'emit("e:a", 1);\nemitNet("e:b");\nemit(`e:${n}`);\nemit("e:" + n, "late");\n'
    'obj.emit("e:o");\n',
    "a.ts": 'emit("e:a", 1);\nemitNet("e:b");\nemit("e:" + n, "late");\nobj.emit("e:o");\n',
    "a.tsx": 'emit("e:a", 1);\nemitNet("e:b");\nemit("e:" + n, "late");\nobj.emit("e:o");\n',
}


@pytest.mark.parametrize("name", sorted(EMITS))
def test_emit_refs_hold_event_and_callee(name: str, parse_text: Parse) -> None:
    refs = [r for r in extract_edges(parse_text(name, EMITS[name])).refs if r.kind == "emit"]
    names = {r.name for r in refs}
    assert {"e:a", "e:b"} <= names
    assert "late" not in names
    assert "e:o" not in names
    assert all(r.qualifier is not None for r in refs)
    callees = {r.name: r.qualifier for r in refs}
    assert callees["e:a"] in {"TriggerServerEvent", "emit"}
    assert callees["e:b"] in {"TriggerClientEvent", "emitNet"}


def test_csharp_static_base_script_emit(parse_text: Parse) -> None:
    refs = extract_edges(parse_text("a.cs", EMITS["a.cs"])).refs
    assert [r.qualifier for r in refs if r.kind == "emit" and r.name == "e:s"] == ["TriggerEvent"]


def test_two_string_arguments_yield_the_first_event(parse_text: Parse) -> None:
    refs = extract_edges(parse_text("a.lua", 'TriggerClientEvent("e:one", "e:two")\n')).refs
    assert [r.name for r in refs if r.kind == "emit"] == ["e:one"]
