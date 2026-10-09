from collections.abc import Callable

from contextweave.lang.extract_symbols import (
    SymbolExtraction,
    extract_symbols,
    preceding_comment_block,
    python_docstring,
)
from contextweave.lang.model import Symbol
from contextweave.lang.parse import ParsedFile, parse_bytes

type Loader = Callable[[str], ParsedFile]
type TextLoader = Callable[[str, str], ParsedFile]


def run(parse_fixture: Loader, name: str) -> tuple[ParsedFile, SymbolExtraction]:
    parsed = parse_fixture(name)
    return parsed, extract_symbols(parsed)


def by_qname(extraction: SymbolExtraction) -> dict[str, Symbol]:
    return {s.qualified_name: s for s in extraction.symbols}


def handlers(extraction: SymbolExtraction) -> list[str]:
    return [s.name for s in extraction.symbols if s.kind == "handler"]


def test_python_kinds_docstrings_and_signature(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.py")
    symbols = by_qname(ex)
    assert set(symbols) == {
        "top_level",
        "Widget",
        "Widget.render",
        "Widget._hidden",
        "undocumented",
    }
    assert symbols["Widget.render"].kind == "method"
    assert symbols["Widget.render"].parent == "Widget"
    assert symbols["top_level"].kind == "function"
    assert symbols["top_level"].docstring == '"""Top-level function doc."""'
    assert symbols["undocumented"].docstring is None
    assert symbols["top_level"].signature == 'def top_level(a: int, b: str = "x") -> bool:'
    imports = {i.module: i for i in ex.imports}
    assert imports["pathlib"].alias == "P"
    assert imports["pathlib"].kind == "from_import"
    assert "Path" in imports["pathlib"].names


def test_python_docstring_helpers(parse_fixture: Loader) -> None:
    parsed = parse_fixture("core/sample.py")
    assert python_docstring(None, parsed.source) is None
    assert preceding_comment_block(parsed.tree.root_node, parsed.source) is None


def test_method_parents_turn_functions_into_methods(parse_fixture: Loader) -> None:
    _, go = run(parse_fixture, "core/sample.ts")
    kinds = {s.qualified_name: s.kind for s in go.symbols}
    assert kinds["Service.run"] == "method"
    assert kinds["build"] == "function"
    _, proto = run(parse_fixture, "data/schema.proto")
    assert by_qname(proto)["UserService.GetUser"].kind == "rpc"


def test_specific_kind_wins_over_general(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.go")
    symbols = by_qname(ex)
    assert symbols["Widget"].kind == "struct"
    assert symbols["Renderer"].kind == "interface"
    assert symbols["Version"].signature == 'const Version = "1.0"'


def test_exports_cover_wrapped_declarations(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.js")
    assert {e.name for e in ex.exports} == {"add", "scale", "Box"}
    assert all(e.profile == "symbols" and e.group == "core" for e in ex.exports)


def test_lua_cfx_handlers_named_by_string(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.lua")
    assert handlers(ex) == ["sample:ping", "sample:ping", "getWidget"]
    first = next(s for s in ex.symbols if s.kind == "handler")
    assert first.signature == 'RegisterNetEvent("sample:ping", function(payload)'
    pings = [s for s in ex.symbols if s.qualified_name == "sample:ping"]
    assert len({s.start_line for s in pings}) == 2


def test_lua_handler_rule_negatives(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.lua")
    names = {s.name for s in ex.symbols}
    assert "sample/select" not in names
    assert not [n for n in names if n.startswith("dyn:")]


def test_lua_cfx_wrappers_and_dynamic_names(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/vrp.lua")
    assert handlers(ex) == [
        "closeMenu",
        "vrp:playerSpawn",
        "vrpcmd",
        "vrp:secure",
        "vrpExport",
    ]
    names = {s.name for s in ex.symbols}
    assert "vrp/select" not in names
    assert not [n for n in names if n.startswith("vrp:") and n.endswith("Name()")]


def test_lua_vrp_loader_and_require_imports(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/vrp.lua")
    assert [(i.kind, i.module) for i in ex.imports] == [
        ("require", "vrp"),
        ("require", "vrp"),
        ("require", "json"),
        ("require", "lib.helper"),
    ]
    assert all("require" not in i.names and "module" not in i.names for i in ex.imports)


def test_lua_dot_colon_and_luals_comment(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.lua")
    symbols = by_qname(ex)
    assert symbols["M.add"].kind == "function"
    assert symbols["M:render"].kind == "method"
    doc = symbols["M.add"].docstring
    assert doc is not None
    assert doc.startswith("--- Adds two numbers.")
    assert "@param" in doc


def test_csharp_mono_script(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.cs")
    symbols = by_qname(ex)
    expected = {
        "ClientMain": "class",
        "ClientMain.OnTick": "method",
        "ClientMain.ClientMain": "method",
        "ClientMain.Health": "property",
        "ClientMain.MaxPlayers": "const",
        "ClientMain.Callback": "type",
        "ClientMain.Point": "struct",
        "IGreeter.Greet": "method",
        "Weather": "enum",
    }
    assert {q: symbols[q].kind for q in expected} == expected


def test_csharp_handlers_wrapped_in_action_and_func(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.cs")
    assert handlers(ex) == ["sample:heartbeat", "getHealth", "heal"]
    heal = next(s for s in ex.symbols if s.name == "heal")
    assert heal.end_line - heal.start_line == 3


def test_csharp_dotnet_handler_forms(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/cfx_net.cs")
    assert handlers(ex) == ["net:bare", "net:cast", "net:wrapped", "netExport", "netCommand"]


def test_csharp_handler_negatives(parse_fixture: Loader) -> None:
    for name in ("core/sample.cs", "core/cfx_net.cs"):
        _, ex = run(parse_fixture, name)
        names = {s.name for s in ex.symbols}
        assert not names & {"onClientResourceStart", "net:named", "net:method"}
        assert not [n for n in names if n.startswith("SELECT")]


def test_csharp_namespaces_do_not_prefix_names(parse_fixture: Loader) -> None:
    _, block = run(parse_fixture, "core/sample.cs")
    _, scoped = run(parse_fixture, "core/modern.cs")
    assert by_qname(block)["ClientMain"].parent is None
    assert by_qname(scoped)["ServerMain"].parent is None


def test_csharp_modern_syntax(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/modern.cs")
    symbols = by_qname(ex)
    assert symbols["PlayerInfo"].kind == "class"
    assert symbols["Vehicle"].kind == "class"
    assert symbols["Coord"].kind == "struct"
    assert symbols["Vehicle.Plate"].kind == "property"
    assert symbols["StringExtensions.IsBlank"].kind == "property"
    assert symbols["StringExtensions.Shout"].kind == "method"
    assert (
        symbols["ServerMain"].signature
        == "public partial class ServerMain(ILogger logger) : BaseScript"
    )
    assert [i.module for i in ex.imports] == ["System.Linq", "CitizenFX.Core"]


def test_csharp_attributes_xml_docs_and_usings(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "core/sample.cs")
    symbols = by_qname(ex)
    assert symbols["ClientMain.OnTick"].signature == "[Tick] public async Task OnTick()"
    assert symbols["ClientMain.OnGreet"].signature.startswith('[EventHandler("sample:greet")]')
    assert (
        symbols["ClientMain.OnClientResourceStart"].docstring
        == "/// <summary>Fires when a resource starts.</summary>"
    )
    imports = {i.module: i for i in ex.imports}
    assert {"System", "CitizenFX.Core.Native.API"} <= set(imports)
    assert imports["System.Collections.Generic"].names == ()
    assert imports["Newtonsoft.Json.JsonConvert"].alias == "Json"


def test_terraform_composite_names(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "devops/main.tf")
    symbols = by_qname(ex)
    assert "aws_s3_bucket.assets" in symbols
    assert symbols["aws_s3_bucket.assets"].docstring == "# Primary bucket."
    assert symbols["region"].signature.startswith('variable "region"')


def test_ruby_comment_climbs_to_ancestor(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "scripting/sample.rb")
    symbols = by_qname(ex)
    assert symbols["Billing.Invoice"].docstring == "# An invoice."
    assert symbols["Billing.Invoice.total"].docstring == "# Returns the total."
    assert symbols["Billing"].signature == "module Billing"


def test_blank_line_detaches_comment(parse_text: TextLoader) -> None:
    parsed = parse_text("a.go", "package m\n\n// detached\n\nfunc F() {}\n")
    assert by_qname(extract_symbols(parsed))["F"].docstring is None


def test_dockerfile_stages_and_edges(parse_fixture: Loader) -> None:
    _, ex = run(parse_fixture, "devops/Dockerfile")
    assert set(by_qname(ex)) == {"build", "runtime"}
    kinds = {(i.kind, i.module) for i in ex.imports}
    assert {("base_image", "node:20-alpine"), ("copy_from", "--from=build")} <= kinds


def test_bash_source_is_import_not_alias(parse_fixture: Loader) -> None:
    for name in ("scripting/sample.sh", "scripting/sample.zsh"):
        _, ex = run(parse_fixture, name)
        assert by_qname(ex)["greet"].docstring == "# Greets the user."
        assert ex.imports[0].module == "./lib.sh"
        assert ex.imports[0].alias is None


def test_byte_ranges_are_authoritative_and_lines_derived(parse_fixture: Loader) -> None:
    for name in ("core/sample.py", "core/sample.cs", "core/vrp.lua", "scripting/sample.rb"):
        parsed, ex = run(parse_fixture, name)
        for symbol in ex.symbols:
            text = parsed.slice(symbol.start_byte, symbol.end_byte)
            assert text.strip() == text
            assert symbol.start_line == parsed.source[: symbol.start_byte].count(b"\n") + 1
            assert symbol.end_line == parsed.source[: symbol.end_byte].count(b"\n") + 1
        for imported in ex.imports:
            assert imported.line == parsed.source[: imported.start_byte].count(b"\n") + 1


def test_every_item_declares_profile_and_group(parse_fixture: Loader) -> None:
    for name in ("core/sample.lua", "devops/main.tf", "web/sample.css", "scripting/sample.sh"):
        parsed, ex = run(parse_fixture, name)
        for item in (*ex.symbols, *ex.imports, *ex.exports):
            assert item.profile == parsed.spec.profile
            assert item.group == parsed.spec.group


def test_syntax_errors_do_not_stop_extraction() -> None:
    result = parse_bytes("broken.py", b"def ok():\n    return 1\n\ndef broken(:\n")
    assert result.parsed is not None
    assert [e.code for e in result.errors] == ["syntax_error"]
    assert "ok" in by_qname(extract_symbols(result.parsed))


def test_schema_profile_has_no_query_file(parse_fixture: Loader) -> None:
    ex = extract_symbols(parse_fixture("data/sample.json"))
    assert [e.code for e in ex.errors] == ["no_query"]
    assert ex.symbols == ()


def test_csharp_local_function_is_symbol(parse_text: TextLoader) -> None:
    ex = extract_symbols(
        parse_text("core/local.cs", "class C { void M() { int Local() { return 1; } } }\n")
    )
    assert by_qname(ex)["C.M.Local"].kind == "function"


def test_go_type_alias_is_symbol(parse_text: TextLoader) -> None:
    ex = extract_symbols(parse_text("core/alias.go", "package p\n\ntype A = int\ntype B int\n"))
    assert {s.name for s in ex.symbols if s.kind == "type"} >= {"A", "B"}
