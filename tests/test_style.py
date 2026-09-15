"""Repository conventions, enforced so they stay true."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
TEXT = (
    PYTHON
    + sorted(ROOT.glob("*.md"))
    + sorted(ROOT.glob("*.toml"))
    + sorted(ROOT.glob("*.cff"))
    + sorted((ROOT / "configs").glob("*.json"))
)
DASHES = (chr(0x2014), chr(0x2013))
BACKTICK = chr(0x60)
TOOL_WORDS = re.compile(
    "|".join(
        ("anthrop" + "ic", "cod" + "ex", "open" + "ai", "chat" + "gpt", "co" + "pilot", "generated " + "with")
    ),
    re.IGNORECASE,
)


def dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def identifiers(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        match node:
            case (
                ast.Name(id=name)
                | ast.FunctionDef(name=name)
                | ast.AsyncFunctionDef(name=name)
                | ast.ClassDef(name=name)
            ):
                names.add(name)
            case ast.arg(arg=name) | ast.keyword(arg=name) if name:
                names.add(name)
            case ast.Attribute(attr=name):
                names.add(name)
            case ast.alias(asname=name) if name:
                names.add(name)
    return names


def test_no_identifier_starts_with_an_underscore() -> None:
    offenders: dict[str, list[str]] = {}
    for path in PYTHON:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bad = sorted(name for name in identifiers(tree) if name.startswith("_") and not dunder(name))
        if bad:
            offenders[str(path.relative_to(ROOT))] = bad
    assert offenders == {}


def test_no_future_imports() -> None:
    for path in PYTHON:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        futures = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ]
        assert futures == [], path


def test_no_dashes_or_backticks_in_text() -> None:
    for path in TEXT:
        text = path.read_text(encoding="utf-8")
        assert not any(dash in text for dash in DASHES), f"{path} contains an em or en dash"
        assert BACKTICK not in text, f"{path} contains a backtick"


def test_no_tooling_attribution() -> None:
    for path in TEXT:
        assert TOOL_WORDS.search(path.read_text(encoding="utf-8")) is None, path


def test_python_files_have_module_docstrings_in_src() -> None:
    for path in (ROOT / "src").rglob("*.py"):
        if path.name == "__main__.py":
            continue
        assert ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))), path
