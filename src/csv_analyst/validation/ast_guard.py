"""Allowlist AST walker for LLM-generated pandas code (PLAN.md Phase 2, CLAUDE.md §8).

Untrusted code from the model is never trusted by shape alone. Every node,
name, and attribute must be explicitly allowed; anything else is rejected
before it ever reaches the execution sandbox. Allowlist over blacklist:
unrecognized nodes are rejected by default, not enumerated as "also bad".
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

MAX_CODE_CHARS = 4000
MAX_CODE_LINES = 40
MAX_AST_DEPTH = 25

ALLOWED_ROOT_NAMES = {"df", "pd", "np"}

# Mirrors execution/sandbox_entry.py's SAFE_BUILTINS (Phase 3) — kept in sync
# so the guard never accepts a call the sandbox would then have no builtin for.
SAFE_BUILTINS = {
    "abs", "min", "max", "sum", "len", "round", "sorted", "range",
    "enumerate", "zip", "list", "dict", "set", "tuple", "float", "int",
    "str", "bool", "isinstance", "any", "all",
}

_ALLOWED_STMT_TYPES = (ast.Assign, ast.AugAssign, ast.Expr)

_ALLOWED_EXPR_TYPES = (
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.Index if hasattr(ast, "Index") else ast.Slice,  # py<3.9 compat no-op
    ast.Call,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    ast.Dict,
    ast.Set,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.Lambda,
    ast.Starred,
    ast.arguments,
    ast.arg,
)

_ALLOWED_OPERATORS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.Invert,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.BitAnd, ast.BitOr, ast.BitXor,
)

# First-hop allowlist for `pd.<attr>` / `np.<attr>`. Review found that
# DENIED_ATTRIBUTES only blocks specific *names*, so a chain like
# `pd.compat.compressors.bz2.os.remove(...)` sails through untouched --
# `compat` isn't denied, and everything after it is an attribute on some
# arbitrary object the guard never inspects. Walking pd/np by attribute
# reached 93 non-pandas modules this way (os, socket, shutil, ctypes,
# importlib, builtins, pickle, zipfile among them), all executing for real
# in the sandbox. Restricting the FIRST hop off `pd`/`np` to known analysis
# entry points closes that path without touching the unbounded, denylist-only
# handling of attributes on `df` or on intermediate call results -- this does
# not make the guard a security boundary, just closes the traversal found above.
ALLOWED_PD_ATTRS = {
    "DataFrame", "Series", "Index", "MultiIndex", "Categorical",
    "Timestamp", "Timedelta", "NA", "NaT",
    "to_datetime", "to_numeric", "to_timedelta", "date_range",
    "isna", "notna", "isnull", "notnull",
    "cut", "qcut", "crosstab", "pivot_table", "get_dummies", "unique",
}
ALLOWED_NP_ATTRS = {
    "nan", "inf", "mean", "median", "std", "var", "sum", "min", "max",
    "abs", "round", "where", "log", "log10", "log2", "sqrt", "exp",
    "ceil", "floor", "percentile", "quantile", "arange", "array", "unique",
    "isnan", "isfinite", "clip", "sign", "maximum", "minimum", "corrcoef",
    "int64", "float64", "bool_",
}

DENIED_ATTRIBUTES = {
    "read_csv", "read_excel", "read_json", "read_parquet", "read_sql", "read_html",
    "read_pickle", "read_feather", "read_orc", "read_stata", "read_gbq", "read_hdf",
    "read_clipboard", "read_table", "read_fwf",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet", "to_hdf",
    "to_feather", "to_orc", "to_stata", "to_gbq", "to_clipboard", "to_markdown", "to_latex",
    "save", "load", "fromfile", "tofile", "dump", "dumps", "loads",
    "eval", "query", "pipe", "merge", "join", "concat",
    "open", "system", "popen", "subprocess", "Popen",
}

DENIED_BUILTINS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
    "breakpoint", "memoryview", "exit", "quit", "help",
}


@dataclass
class GuardResult:
    ok: bool
    reason: str | None = None
    warnings: list[str] = field(default_factory=list)


def validate_code(code: str) -> GuardResult:
    """Reject unsafe generated code. Returns a machine-readable reason on
    failure, meant to be fed back to the model on retry."""

    if len(code) > MAX_CODE_CHARS:
        return GuardResult(ok=False, reason=f"code exceeds {MAX_CODE_CHARS} characters")
    if code.count("\n") + 1 > MAX_CODE_LINES:
        return GuardResult(ok=False, reason=f"code exceeds {MAX_CODE_LINES} lines")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return GuardResult(ok=False, reason=f"syntax error: {exc}")

    depth = _max_depth(tree)
    if depth > MAX_AST_DEPTH:
        return GuardResult(ok=False, reason=f"AST depth {depth} exceeds {MAX_AST_DEPTH}")

    assigned_names: set[str] = set()
    result_assigned = False

    for stmt in tree.body:
        if not isinstance(stmt, _ALLOWED_STMT_TYPES):
            return GuardResult(ok=False, reason=f"disallowed statement: {type(stmt).__name__}")

        error = _check_node(stmt, assigned_names)
        if error:
            return GuardResult(ok=False, reason=error)

        for target in _assignment_targets(stmt):
            assigned_names.add(target)
            if target == "result":
                result_assigned = True

    if not result_assigned:
        return GuardResult(ok=False, reason="code must assign to a variable named 'result'")

    # A snippet that never reads `df` can still assign a bare literal to
    # `result` (e.g. `result = 999999`) and pass every check above: syntax is
    # valid, only `result` is assigned, no denied names/attributes appear.
    # That snippet then executes "successfully" and produces a number that
    # was never derived from the dataset -- exactly the fabrication this
    # guard exists to prevent. Require at least one Load of `df` so a
    # constant-only "analysis" is rejected before it ever reaches the sandbox.
    reads_df = any(
        isinstance(n, ast.Name) and n.id == "df" and isinstance(n.ctx, ast.Load)
        for n in ast.walk(tree)
    )
    if not reads_df:
        return GuardResult(
            ok=False,
            reason="code must read from the dataframe 'df' -- a result not derived "
                   "from the data is not an answer",
        )

    return GuardResult(ok=True)


def _assignment_targets(stmt: ast.stmt) -> list[str]:
    names: list[str] = []
    targets: list[ast.expr] = []
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, ast.AugAssign):
        targets = [stmt.target]

    for t in targets:
        for node in ast.walk(t):
            if isinstance(node, ast.Name):
                names.append(node.id)
    return names


def _locally_bound_names(node: ast.AST) -> set[str]:
    """Names bound within this statement by comprehension targets or lambda
    params — scoped to the statement, so a Name Load inside a comprehension
    or lambda body may legally reference them even though no top-level
    Assign put them in `assigned_names`."""
    bound: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.comprehension):
            bound.update(n.id for n in ast.walk(child.target) if isinstance(n, ast.Name))
        if isinstance(child, ast.Lambda):
            bound.update(a.arg for a in child.args.args)
            bound.update(a.arg for a in child.args.posonlyargs)
            bound.update(a.arg for a in child.args.kwonlyargs)
    return bound


def _check_node(node: ast.AST, assigned_names: set[str]) -> str | None:
    lambda_error = _check_lambda_positions(node)
    if lambda_error:
        return lambda_error

    local_names = assigned_names | ALLOWED_ROOT_NAMES | SAFE_BUILTINS | _locally_bound_names(node)

    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and child is not node and not isinstance(child, _ALLOWED_STMT_TYPES):
            return f"disallowed statement: {type(child).__name__}"

        if isinstance(child, (ast.Import, ast.ImportFrom)):
            return "import statements are not allowed"

        if isinstance(child, (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef, ast.With, ast.AsyncWith, ast.Try, ast.Global,
                               ast.Nonlocal, ast.Delete, ast.Assert, ast.Raise, ast.Yield,
                               ast.YieldFrom, ast.Await)):
            return f"disallowed construct: {type(child).__name__}"

        if isinstance(child, ast.Name):
            if child.id.startswith("_"):
                return f"disallowed identifier: {child.id}"
            if child.id in DENIED_BUILTINS:
                return f"disallowed builtin: {child.id}"
            if isinstance(child.ctx, ast.Load) and child.id not in local_names:
                return f"unknown name: {child.id}"

        if isinstance(child, ast.Attribute):
            if child.attr.startswith("_"):
                return f"disallowed attribute: {child.attr}"
            if child.attr in DENIED_ATTRIBUTES:
                return f"disallowed attribute: {child.attr}"
            if isinstance(child.value, ast.Name) and child.value.id == "pd" \
                    and child.attr not in ALLOWED_PD_ATTRS:
                return f"pd.{child.attr} is not an allowed analysis function"
            if isinstance(child.value, ast.Name) and child.value.id == "np" \
                    and child.attr not in ALLOWED_NP_ATTRS:
                return f"np.{child.attr} is not an allowed analysis function"

        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                and child.func.id in DENIED_BUILTINS:
            return f"disallowed call: {child.func.id}"

        if isinstance(child, (ast.BinOp,)) and not isinstance(child.op, _ALLOWED_OPERATORS):
            return f"disallowed operator: {type(child.op).__name__}"
        if isinstance(child, (ast.UnaryOp,)) and not isinstance(child.op, _ALLOWED_OPERATORS):
            return f"disallowed operator: {type(child.op).__name__}"
        if isinstance(child, (ast.BoolOp,)) and not isinstance(child.op, _ALLOWED_OPERATORS):
            return f"disallowed operator: {type(child.op).__name__}"
        if isinstance(child, ast.Compare):
            for op in child.ops:
                if not isinstance(op, _ALLOWED_OPERATORS):
                    return f"disallowed comparator: {type(op).__name__}"

        if not isinstance(child, _ALLOWED_EXPR_TYPES + _ALLOWED_STMT_TYPES + _ALLOWED_OPERATORS
                          + (ast.Module, ast.operator, ast.unaryop, ast.boolop, ast.cmpop, ast.expr_context)):
            return f"disallowed node: {type(child).__name__}"

    return None


def _check_lambda_positions(node: ast.AST) -> str | None:
    """Lambda is allowed only as a call argument (.apply/.map style), never
    bound to a name via Assign/AugAssign."""
    parent_of: dict[int, ast.AST] = {}
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            parent_of[id(child)] = parent

    for lam in ast.walk(node):
        if not isinstance(lam, ast.Lambda):
            continue
        parent = parent_of.get(id(lam))
        if not isinstance(parent, ast.Call):
            return "lambda is only allowed as a call argument"
    return None


def _max_depth(node: ast.AST, current: int = 0) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return current
    return max(_max_depth(child, current + 1) for child in children)
