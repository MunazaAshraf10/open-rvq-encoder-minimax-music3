import ast
import re
from pathlib import Path

import rvq_ae

ROOT = Path(__file__).resolve().parents[1]
PYTHON = (
    sorted((ROOT / "src").rglob("*.py"))
    + sorted((ROOT / "tests").rglob("*.py"))
    + sorted((ROOT / "paper").glob("*.py"))
)
TEXT = (
    PYTHON
    + sorted(ROOT.glob("*.md"))
    + sorted(ROOT.glob("*.toml"))
    + sorted(ROOT.glob("*.cff"))
    + sorted((ROOT / "configs").glob("*.json"))
    + sorted((ROOT / "benchmarks").glob("*.md"))
    + sorted((ROOT / "docs").glob("*.md"))
    + sorted((ROOT / "results").rglob("*.md"))
    + sorted((ROOT / "paper").rglob("*.tex"))
    + sorted((ROOT / "paper").rglob("*.bib"))
    + sorted((ROOT / "paper").glob("*.md"))
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


def test_no_module_docstrings() -> None:
    """Documentation belongs on the class or function it describes, never above the imports."""
    for path in PYTHON:
        docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        assert docstring is None, f"{path} opens with a module docstring"


def test_public_api_is_exported() -> None:
    """Every name in rvq_ae.__all__ resolves, so the package surface stays honest."""
    for name in rvq_ae.__all__:
        assert hasattr(rvq_ae, name), name
    assert rvq_ae.__all__ == sorted(rvq_ae.__all__)


def test_imports_are_at_the_top_of_the_file() -> None:
    """Only the optional extras may be imported lazily, so the base install stays importable."""
    optional = {"wandb", "uvicorn"}
    offenders: dict[str, list[str]] = {}
    for path in PYTHON:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top = {id(node) for node in tree.body}
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Import | ast.ImportFrom) or id(node) in top:
                continue
            root = (getattr(node, "module", None) or node.names[0].name).split(".")[0]
            if root not in optional:
                bad.append(f"{node.lineno}: {root}")
        if bad:
            offenders[str(path.relative_to(ROOT))] = bad
    assert offenders == {}
