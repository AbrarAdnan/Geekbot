"""AST guard tests (PLAN.md Phase 2 exit gate): a red-team corpus of >= 30
malicious/unsafe snippets must all be rejected (zero false negatives), and
>= 15 legitimate analysis snippets must all be accepted."""

from __future__ import annotations

import pytest

from csv_analyst.validation.ast_guard import validate_code

RED_TEAM_SNIPPETS = [
    "import os\nresult = 1",
    "__import__('os').system('ls')\nresult = 1",
    "result = ().__class__.__bases__",
    "result = [].__class__.__mro__",
    "result = df.__class__.__init__.__globals__",
    "result = pd.read_csv('/etc/passwd')",
    "df.to_csv('out.csv')\nresult = 1",
    "result = open('secret.txt').read()",
    "result = eval('1+1')",
    "result = exec('1+1')",
    "result = compile('1+1', '<s>', 'eval')",
    "result = df.query('@pd')",
    "result = df.merge(df)",
    "result = df.join(df)",
    "result = pd.concat([df, df])",
    "import subprocess\nresult = 1",
    "result = getattr(df, 'to_csv')",
    "result = globals()",
    "result = locals()",
    "result = vars()",
    "result = dir(df)",
    "result = __builtins__",
    "for i in range(10):\n    result = i",
    "while True:\n    result = 1\n    break",
    "def f():\n    return 1\nresult = f()",
    "class Foo:\n    pass\nresult = 1",
    "with open('x') as f:\n    result = 1",
    "try:\n    result = 1\nexcept Exception:\n    result = 2",
    "assert True\nresult = 1",
    "raise ValueError()\nresult = 1",
    "global x\nresult = 1",
    "del df\nresult = 1",
    "result = 1\nyield result",
    "f = lambda: os.system('ls')\nresult = f()",
    "result = df.apply(lambda x: x.__class__)",
    "result = df['a'].values.tofile('x.bin')",
    "result = df.to_pickle('x.pkl')",
    "result = df.to_sql('t', None)",
    "result = os.popen('ls').read()",
    "result = 1; import sys",
    "result = df\nprint(result)",
    "x = input()\nresult = x",
    "result = breakpoint()",
    "result = memoryview(b'x')",
    "result = 999999",
    "result = {'total': 999999}",
    "result = [1, 2, 3]",
    "result = pd.DataFrame({'a': [1, 2]})",
    "result = pd.compat.compressors.bz2.os.remove('x')\nresult = df",
    "result = pd.io.common.os.listdir('.')\nresult = df",
    "result = pd.compat.pyarrow.pa.util.socket.socket()\nresult = df",
    "result = np.core.os.getcwd()\nresult = df",
    "result = pd.compat.compressors.lzma.builtins.compile('1','<s>','eval')\nresult = df",
    "result = pd.io.common.tarfile.shutil.disk_usage('C:/')\nresult = df",
]

LEGIT_SNIPPETS = [
    "result = df['revenue'].mean()",
    "result = df.groupby('region')['revenue'].mean().reset_index()",
    "result = df[df['region'] == 'North']",
    "result = df.sort_values('revenue', ascending=False).head(10)",
    "result = len(df)",
    "result = df['revenue'].sum()",
    "result = df.isna().sum()",
    "result = df['revenue'].fillna(0).mean()",
    "result = df.groupby(['region', 'order_date'])['units'].sum()",
    "result = df[(df['revenue'] > 100) & (df['units'] < 5)]",
    "mask = df['notes'].isna()\nresult = df[mask]",
    "result = df['revenue'].apply(lambda x: x * 1.1)",
    "result = df.assign(margin=df['revenue'] - df['units'])",
    "counts = df['region'].value_counts()\nresult = counts",
    "result = df.pivot_table(index='region', values='revenue', aggfunc='mean')",
    "result = df['order_date'].max()",
    "total = df['revenue'].sum()\navg = df['revenue'].mean()\nresult = {'total': total, 'avg': avg}",
    "result = df.nlargest(5, 'revenue')",
    "result = df['revenue'].std() if len(df) > 1 else 0.0",
    "result = [r for r in df['region'].unique()]",
    "result = pd.to_datetime(df['order_date']).dt.year.value_counts()",
    "result = df.assign(flag=np.where(df['revenue'] > 100, 1, 0))['flag'].sum()",
    "result = pd.DataFrame({'x': df['revenue'].describe()}).reset_index()",
]


@pytest.mark.parametrize("code", RED_TEAM_SNIPPETS)
def test_red_team_snippet_rejected(code):
    outcome = validate_code(code)
    assert outcome.ok is False, f"should have been rejected: {code!r}"
    assert outcome.reason


@pytest.mark.parametrize("code", LEGIT_SNIPPETS)
def test_legit_snippet_accepted(code):
    outcome = validate_code(code)
    assert outcome.ok is True, f"should have been accepted: {code!r} -> {outcome.reason}"


def test_missing_result_assignment_rejected():
    outcome = validate_code("x = df['revenue'].mean()")
    assert outcome.ok is False
    assert "result" in outcome.reason


def test_oversized_code_rejected():
    code = "result = " + " + ".join(["1"] * 3000)
    outcome = validate_code(code)
    assert outcome.ok is False


def test_too_many_lines_rejected():
    code = "\n".join([f"x{i} = {i}" for i in range(45)]) + "\nresult = 1"
    outcome = validate_code(code)
    assert outcome.ok is False


def test_syntax_error_rejected():
    outcome = validate_code("result = df[[")
    assert outcome.ok is False
    assert "syntax" in outcome.reason
