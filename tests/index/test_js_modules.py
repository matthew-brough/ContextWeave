from pathlib import Path

from edge_support import build_edges

PROJECT = {
    "app/package.json": '{"name": "app", "main": "./lib/entry.js"}',
    "app/lib/entry.js": "export const entry = 1;\n",
    "app/cjs.js": 'const { entry } = require("./");\nmodule.exports = { entry };\n',
    "app/esm.mjs": 'import { entry } from ".";\nexport { entry };\n',
    "app/reexport.js": 'export * from "./lib/entry.js";\n',
    "app/lazy.js": 'const load = () => import("./lib/entry.js");\n',
    "pkgs/kit/package.json": (
        '{"name": "@x/kit", "exports": {".": {"import": "./src/m.js", "require": "./src/c.js"},'
        ' "./extra": "./src/extra.js"}}'
    ),
    "pkgs/kit/src/m.js": "export const m = 1;\n",
    "pkgs/kit/src/c.js": "exports.m = 1;\n",
    "pkgs/kit/src/extra.js": "export const e = 1;\n",
    "pkgs/old/package.json": '{"name": "old", "main": "lib/start.js"}',
    "pkgs/old/lib/start.js": "exports.s = 1;\n",
    "use.js": (
        'import { m } from "@x/kit";\nimport { e } from "@x/kit/extra";\n'
        'const old = require("old");\n'
    ),
}


def test_cjs_esm_reexport_and_dynamic_resolve_to_the_same_files(tmp_path: Path) -> None:
    edges = build_edges(tmp_path, PROJECT)
    assert ("app/cjs.js", "imports", "app/lib/entry.js", "exact") in edges
    assert ("app/esm.mjs", "imports", "app/lib/entry.js", "exact") in edges
    assert ("app/reexport.js", "imports", "app/lib/entry.js", "exact") in edges
    assert ("app/lazy.js", "imports", "app/lib/entry.js", "exact") in edges


def test_package_exports_and_main(tmp_path: Path) -> None:
    edges = build_edges(tmp_path, PROJECT)
    assert ("use.js", "imports", "pkgs/kit/src/m.js", "exact") in edges
    assert ("use.js", "imports", "pkgs/kit/src/extra.js", "exact") in edges
    assert ("use.js", "imports", "pkgs/old/lib/start.js", "exact") in edges
    assert ("use.js", "imports", "pkgs/kit/src/c.js", "exact") not in edges
