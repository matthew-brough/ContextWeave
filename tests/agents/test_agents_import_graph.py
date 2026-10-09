import ast
from pathlib import Path

import contextweave

PACKAGE = Path(contextweave.__file__).parent
AGNOSTIC = ("core", "index", "retrieve", "compact", "memory", "models")
FORBIDDEN = "contextweave.agents"


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_agent_agnostic_packages_never_import_agents() -> None:
    offenders: list[str] = []
    checked = 0
    for package in AGNOSTIC:
        for path in sorted((PACKAGE / package).rglob("*.py")):
            checked += 1
            for name in imported_modules(path):
                if name == FORBIDDEN or name.startswith(FORBIDDEN + "."):
                    offenders.append(f"{path.relative_to(PACKAGE)}: {name}")
    assert checked > 20
    assert offenders == []
