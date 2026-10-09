import json
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from contextweave.core.config import Config, EventsConfig
from contextweave.index.edges import resolve_edges
from contextweave.index.events import is_net, parse_manifest, target_sides
from contextweave.index.indexer import index_project
from contextweave.index.store import Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fivem"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "events" / "fivem.json"

EMITS = """
SELECT ff.path, e.line, COALESCE(fs.qualified_name, ''), ts.qualified_name, tf.path, e.detail
FROM symbol_edges e
LEFT JOIN symbols fs ON fs.id = e.from_id
JOIN files ff ON ff.id = e.from_file_id
JOIN symbols ts ON ts.id = e.to_id
JOIN files tf ON tf.id = e.to_file_id
WHERE e.rel = 'emits'
ORDER BY 1, 2, 3, 4, 5
"""


@dataclass(frozen=True, slots=True)
class Emit:
    source: str
    line: int
    emitter: str
    handler: str
    target: str
    event: str
    net: bool
    sides_from: tuple[str, ...]
    sides_to: tuple[str, ...]

    def pair(self) -> tuple[str, str]:
        return (self.source, self.target)

    def wire(self) -> dict[str, str | int | bool | list[str]]:
        return {
            "source": self.source,
            "line": self.line,
            "emitter": self.emitter,
            "handler": self.handler,
            "target": self.target,
            "event": self.event,
            "net": self.net,
            "from": list(self.sides_from),
            "to": list(self.sides_to),
        }


def read_emits(root: Path) -> list[Emit]:
    found: list[Emit] = []
    with Store.open(root) as store:
        for row in store.query(EMITS):
            detail = json.loads(str(row[5]))
            found.append(
                Emit(
                    str(row[0]),
                    int(str(row[1])),
                    str(row[2]),
                    str(row[3]),
                    str(row[4]),
                    detail["event"],
                    detail["net"],
                    tuple(detail["sides"]["from"]),
                    tuple(detail["sides"]["to"]),
                )
            )
    return found


@pytest.fixture(scope="module")
def emits(tmp_path_factory: pytest.TempPathFactory) -> Iterator[list[Emit]]:
    root = tmp_path_factory.mktemp("fivem")
    shutil.copytree(FIXTURES, root, dirs_exist_ok=True)
    (root / ".git").mkdir()
    index_project(root, Config())
    yield read_emits(root)


def between(emits: list[Emit], source: str, target: str, event: str | None = None) -> list[Emit]:
    return [
        e
        for e in emits
        if e.source == source and e.target == target and (event is None or e.event == event)
    ]


def test_matches_golden(emits: list[Emit]) -> None:
    actual = [e.wire() for e in emits]
    if os.environ.get("CW_UPDATE_GOLDEN"):
        GOLDEN.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    assert actual == json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_lua_reaches_csharp_and_back(emits: list[Emit]) -> None:
    forward = between(emits, "duel/client/main.lua", "duel/server/Server.cs", "duel:csServer")
    assert [(e.net, e.sides_from, e.sides_to) for e in forward] == [
        (True, ("client",), ("server",))
    ]
    assert forward[0].handler.endswith("duel:csServer")
    back = between(emits, "duel/server/main.lua", "duel/client/Client.cs", "duel:csClient")
    assert len(back) == 1
    assert between(emits, "duel/server/Server.cs", "duel/client/main.lua", "duel:clientPing")
    assert between(emits, "duel/client/Client.cs", "duel/server/main.lua", "duel:serverHello")


def test_latent_variants_link_like_plain_ones(emits: list[Emit]) -> None:
    bulk = between(emits, "duel/client/main.lua", "duel/server/main.lua", "duel:bulk")
    assert len(bulk) == 1
    assert bulk[0].net


def test_server_emit_never_reaches_client_handler_and_plain_stays_on_side(
    emits: list[Emit],
) -> None:
    both = [e for e in emits if e.event == "duel:both"]
    plain_client = [e for e in both if e.source == "duel/client/main.lua" and not e.net]
    assert {e.target for e in plain_client} == {"duel/client/main.lua", "duel/tools/orphan.lua"}
    server_net = [e for e in both if e.source == "duel/client/main.lua" and e.net]
    assert {e.target for e in server_net} == {"duel/server/main.lua"}
    from_server = [e for e in both if e.source == "duel/server/main.lua"]
    assert {(e.target, e.net) for e in from_server} == {
        ("duel/client/main.lua", True),
        ("duel/tools/orphan.lua", True),
        ("duel/server/main.lua", False),
    }


def test_shared_handler_is_reached_from_both_sides(emits: list[Emit]) -> None:
    assert between(emits, "duel/client/main.lua", "duel/shared.lua", "duel:sharedHook")
    assert between(emits, "duel/server/main.lua", "duel/shared.lua", "duel:sharedHook")


def test_file_without_manifest_coverage_is_not_filtered(emits: list[Emit]) -> None:
    orphan_in = between(emits, "duel/server/main.lua", "duel/tools/orphan.lua", "duel:both")
    assert orphan_in
    assert orphan_in[0].sides_to == ()
    orphan_out = between(emits, "duel/tools/orphan.lua", "duel/client/main.lua")
    assert orphan_out
    assert orphan_out[0].sides_from == ()


def test_concatenated_names_make_no_link(emits: list[Emit]) -> None:
    assert not [e for e in emits if "dynamic" in e.handler or "dynamic" in e.event]
    assert not [e for e in emits if e.source == "node16/server.js" and e.line == 12]


def test_callee_must_be_a_plain_identifier(emits: list[Emit]) -> None:
    assert not between(emits, "node16/client.js", "node16/server.js", "n16:pong")
    server_ping = [e for e in emits if e.event == "n16:ping"]
    assert {e.target for e in server_ping} == {"node16/server.js"}


def test_node16_cjs(emits: list[Emit]) -> None:
    assert between(emits, "node16/server.js", "node16/client.js", "n16:pong")
    assert between(emits, "node16/client.js", "node16/server.js", "n16:ping")
    assert between(emits, "node16/server.js", "node16/server.js", "n16:local")
    assert not between(emits, "node16/server.js", "node16/client.js", "n16:local")


def test_emit_net_flips_side(emits: list[Emit]) -> None:
    flipped = between(emits, "node22/client.js", "node22/server.js", "n22:ping")
    assert [(e.net, e.sides_from, e.sides_to) for e in flipped] == [
        (True, ("client",), ("server",))
    ]
    assert between(emits, "node22/server.js", "node22/client.js", "n22:pong")
    assert not between(emits, "node22/client.js", "node22/server.js", "n22:tick")


def test_node22_and_node26_esm(emits: list[Emit]) -> None:
    assert between(emits, "node22/server.js", "duel/client/Client.cs", "duel:csClient")
    assert between(emits, "node22/server.js", "duel/shared.lua", "duel:sharedHook")
    assert between(emits, "node26/server.js", "node26/client.js", "n26:pong")
    assert between(emits, "node26/client.js", "node26/server.js", "n26:ping")
    assert between(emits, "node26/server.js", "node26/lib/handlers.js", "n26:late")


def test_client_v8_handlers(emits: list[Emit]) -> None:
    ready = between(emits, "v8client/client/main.js", "v8client/client/main.js", "v8:ready")
    assert [e.emitter for e in ready] == ["v8cmd"]
    assert not [e for e in emits if e.event in {"v8:net", "v8export", "v8cmd"}]


def test_typescript_and_tsx(emits: list[Emit]) -> None:
    assert between(emits, "tsres/client/main.ts", "tsres/server/main.ts", "ts:ping")
    assert between(emits, "tsres/server/main.ts", "tsres/client/main.ts", "ts:pong")
    assert between(emits, "tsres/client/main.ts", "tsres/client/ui.tsx", "ts:ui")
    assert not between(emits, "tsres/client/main.ts", "tsres/server/main.ts", "ts:tick")


def test_every_edge_is_event_resolution(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES, tmp_path, dirs_exist_ok=True)
    (tmp_path / ".git").mkdir()
    index_project(tmp_path, Config())
    with Store.open(tmp_path) as store:
        rows = store.query("SELECT DISTINCT resolution FROM symbol_edges WHERE rel = 'emits'")
    assert [str(row[0]) for row in rows] == ["event"]


def test_project_wrapper_registrar_and_emitter_append(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.lua").write_text(
        'RegisterSecureEvent("w:go", function() end)\nfunction f() TriggerSecure("w:go") end\n',
        encoding="utf-8",
    )
    index_project(tmp_path, Config())
    assert read_emits(tmp_path) == []
    wrapped = EventsConfig(registrars=("RegisterSecureEvent",), emitters=("TriggerSecure",))
    with Store.open(tmp_path) as store:
        resolve_edges(store, (), force=True, events=wrapped)
    assert [e.event for e in read_emits(tmp_path)] == ["w:go"]


def test_manifest_forms() -> None:
    parsed = parse_manifest(
        "-- server_script 'no.lua'\n"
        "--[[ client_script 'no2.lua' ]]\n"
        "client_scripts { 'a.lua', \"b/*.lua\", '@lib/x.lua' }\n"
        "client_script('c.lua')\n"
        "server_script 'd.lua'\n"
        "shared_scripts {\n  './e.lua',\n}\n"
    )
    assert parsed == {
        "client": ("a.lua", "b/*.lua", "c.lua"),
        "server": ("d.lua",),
        "shared": ("e.lua",),
    }


@pytest.mark.parametrize(
    ("callee", "source", "expected"),
    [
        ("TriggerServerEvent", frozenset({"client"}), frozenset({"server", "shared"})),
        ("TriggerLatentClientEvent", frozenset({"server"}), frozenset({"client", "shared"})),
        ("TriggerEvent", frozenset({"client"}), frozenset({"client", "shared"})),
        ("TriggerEvent", frozenset[str](), None),
        ("TriggerEvent", frozenset({"shared"}), None),
        ("emit", frozenset({"server"}), frozenset({"server", "shared"})),
        ("emitNet", frozenset({"client"}), frozenset({"server", "shared"})),
        ("emitNet", frozenset({"server"}), frozenset({"client", "shared"})),
        ("emitNet", frozenset[str](), None),
        ("emitNet", frozenset({"shared"}), None),
    ],
)
def test_target_sides(callee: str, source: frozenset[str], expected: frozenset[str] | None) -> None:
    assert target_sides(callee, source) == expected


def test_net_flag() -> None:
    assert is_net("emitNet")
    assert is_net("TriggerLatentServerEvent")
    assert not is_net("TriggerEvent")
    assert not is_net("emit")
