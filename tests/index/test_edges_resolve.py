from pathlib import Path

from edge_support import build_edges

PYTHON_PROJECT = {
    "pkg/__init__.py": "",
    "pkg/base.py": "class Base:\n    def hello(self) -> str:\n        return 'x'\n",
    "pkg/util.py": "def helper() -> int:\n    return 1\n",
    "pkg/svc.py": (
        "from .base import Base\n"
        "from pkg import util\n\n\n"
        "class Svc(Base):\n"
        "    def run(self) -> int:\n"
        "        self.hello()\n"
        "        return util.helper()\n"
    ),
    "tests/test_svc.py": (
        "from pkg.svc import Svc\n\n\ndef test_run() -> None:\n    Svc().run()\n"
    ),
    "settings.json": '{"limits": {"max_items": 3}}\n',
    "pkg/conf.py": "def limit(cfg: dict[str, int]) -> int:\n    return cfg['limits.max_items']\n",
}


def test_python_imports_calls_inheritance(tmp_path: Path) -> None:
    edges = build_edges(tmp_path, PYTHON_PROJECT)
    assert ("pkg/svc.py", "imports", "pkg/base.py", "exact") in edges
    assert ("pkg/svc.py", "imports", "pkg/util.py", "import") in edges
    assert ("Svc", "inherits", "Base", "exact") in edges
    assert ("Svc.run", "calls", "Base.hello", "exact") in edges
    assert ("Svc.run", "calls", "helper", "exact") in edges


def test_python_tests_point_at_subject(tmp_path: Path) -> None:
    edges = build_edges(tmp_path, PYTHON_PROJECT)
    assert ("test_run", "tests", "Svc", "exact") in edges
    assert ("tests/test_svc.py", "tests", "pkg/svc.py", "exact") in edges


def test_config_edges_are_name_resolution(tmp_path: Path) -> None:
    edges = build_edges(tmp_path, PYTHON_PROJECT)
    config = {edge for edge in edges if edge[1] == "config"}
    assert config
    assert {edge[3] for edge in config} == {"name"}
    assert any(edge[0] == "limit" and "limits.max_items" in edge[2] for edge in config)


def test_unresolved_calls_and_imports_are_dropped(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {"a.py": "import os\nimport requests\n\n\ndef f() -> None:\n    requests.get('x')\n"},
    )
    assert not {edge for edge in edges if edge[1] in {"calls", "imports"}}


def test_unresolved_base_is_kept_with_name_resolution(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path, {"a.py": "from lib import Thing\n\n\nclass A(Thing):\n    pass\n"}
    )
    assert ("A", "inherits", "Thing", "name") in edges


def test_javascript_and_typescript_imports_and_tests(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "src/a.js": "import { b } from './b.js';\nexport function a(){ return b(); }\n",
            "src/b.ts": "export function b(){ return 1; }\n",
            "test/a.test.js": "import {a} from '../src/a.js';\nfunction t(){ a(); }\n",
        },
    )
    assert ("src/a.js", "imports", "src/b.ts", "exact") in edges
    assert ("a", "calls", "b", "exact") in edges
    assert ("t", "tests", "a", "exact") in edges


def test_tsconfig_path_alias_resolves(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "tsconfig.json": '{"compilerOptions": {"baseUrl": ".", "paths": {"@lib/*": ["lib/*"]}}}',
            "lib/x.ts": "export function x(): number { return 1; }\n",
            "app.ts": "import { x } from '@lib/x';\nexport function go(): number { return x(); }\n",
        },
    )
    assert ("app.ts", "imports", "lib/x.ts", "exact") in edges
    assert ("go", "calls", "x", "exact") in edges


def test_go_siblings_share_package(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "go.mod": "module ex.com/m\n\ngo 1.22\n",
            "main.go": "package main\n\nfunc main() { Helper() }\n",
            "h.go": "package main\n\nfunc Helper() {}\n",
        },
    )
    assert ("main", "calls", "Helper", "exact") in edges


def test_go_module_import_resolves_to_package_files(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "go.mod": "module ex.com/m\n\ngo 1.22\n",
            "main.go": 'package main\n\nimport "ex.com/m/util"\n\nfunc main() { util.Do() }\n',
            "util/u.go": "package util\n\nfunc Do() {}\n",
        },
    )
    assert ("main.go", "imports", "util/u.go", "exact") in edges
    assert ("main", "calls", "Do", "exact") in edges


def test_lua_require_resolves_dotted_module(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "a.lua": "local b = require('lib.b')\nlocal function f() return b.g() end\n",
            "lib/b.lua": "local M = {}\nfunction M.g() return 1 end\nreturn M\n",
        },
    )
    assert ("a.lua", "imports", "lib/b.lua", "exact") in edges
    assert ("f", "calls", "M.g", "exact") in edges


def test_lua_table_member_calls_resolve(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "base.lua": (
                "vRP = {}\nfunction vRP.log(text) print(text) end\n"
                "function vRP:save() end\nfunction log() end\n"
            ),
            "user.lua": "local function pay() vRP.log('x') vRP:save() end\n",
        },
    )
    assert ("pay", "calls", "vRP.log", "exact") in edges
    assert ("pay", "calls", "vRP:save", "exact") in edges
    assert ("pay", "calls", "log", "exact") not in edges


def test_csharp_base_list_splits_class_and_interface(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "A.cs": (
                "namespace N { public interface IFoo { } public class Base { public void Hi(){} }"
                " public class D : Base, IFoo { public void Go(){ this.Hi(); } } }\n"
            )
        },
    )
    assert ("D", "inherits", "Base", "exact") in edges
    assert ("D", "implements", "IFoo", "exact") in edges
    assert ("D.Go", "calls", "Base.Hi", "exact") in edges


def test_csharp_using_namespace_links_files(tmp_path: Path) -> None:
    edges = build_edges(
        tmp_path,
        {
            "Lib.cs": "namespace Lib { public class Tool { public void Run(){} } }\n",
            "App.cs": (
                "using Lib;\nnamespace App { public class P { public void Go(){ new Tool().Run(); } } }\n"
            ),
        },
    )
    assert ("App.cs", "imports", "Lib.cs", "exact") in edges


def test_reindex_without_changes_keeps_edges(tmp_path: Path) -> None:
    first = build_edges(tmp_path, PYTHON_PROJECT)
    second = build_edges(tmp_path, PYTHON_PROJECT)
    assert first == second
