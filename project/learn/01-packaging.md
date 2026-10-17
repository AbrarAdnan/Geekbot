# 01 — Python packaging: src layout, pyproject, entry points, conda vs pip, lockfiles, quality tools

**Step:** S0.1 (Skeleton + conda env) · **Written:** 2026-10-07

## Why this step comes first
Before any RAG code exists, the project needs a home that anyone can rebuild exactly: you on a fresh PC, a reviewer, or CI (the automatic checks on GitHub, step S0.5). Most "works on my machine" bugs come from skipping this.

v1 shows what happens otherwise (PLAN.md §5 #16): its `requirements.txt` is a raw `pip freeze` of one machine. It pins 130+ packages, misses packages the code really needs (`unstructured`, `jq`), and has no tests, lint or packaging. This step replaces it with four things:
1. a package layout
2. one config file (`pyproject.toml`)
3. a reproducible environment (`environment.yml` + a lock file)
4. automatic quality checks (ruff, mypy, pytest, pre-commit)

---

## 1. The src layout
```
Geekbot/
  pyproject.toml
  src/
    geekbot/
      __init__.py
  tests/
    unit/test_smoke.py
```
The alternative is the **flat layout**: the `geekbot/` folder sits directly in the repo root.

**The problem with flat:** when you run Python or pytest from the repo root, that folder is on `sys.path` (the list of places Python imports from). So `import geekbot` silently picks up the folder next to you, even if the package was never installed, or installs broken (a missing dependency, a data file left out of the build). Tests pass on your machine and fail everywhere else.

**What src changes:** `src/` is *not* on `sys.path`, so the only way to import `geekbot` is to **install** it. Tests then exercise the same thing a user gets.

**The cost:** you must install the project before tests can run. That's one line in `environment.yml`.

### Editable installs
`pip install -e .` ("editable") doesn't copy the code into the environment. With our build backend (hatchling) it writes a tiny `.pth` file into `site-packages` that adds `src/` to `sys.path`. Your edits are live without reinstalling.

You only re-install when the **metadata** in `pyproject.toml` changes: a new dependency, a new command-line entry point, a new version.

The smoke test `test_imported_from_src` checks exactly this: that `geekbot` is imported from `src/`, not from a stale copy somewhere else.

---

## 2. `pyproject.toml`: one file, three jobs

| Section | Standard | Job |
|---|---|---|
| `[build-system]` | PEP 517 / 518 | Names the tool that *builds* the package (the "build backend"). pip creates a temporary isolated environment, installs the backend into it, and asks it to build. |
| `[project]` | PEP 621 | Standard metadata: name, version, Python versions, dependencies, extras, scripts, license. |
| `[tool.*]` | per tool | Settings for ruff, mypy, pytest, all in one file (instead of `setup.cfg`, `.flake8`, `mypy.ini`, `pytest.ini`). |

**Build backend: hatchling.** Small config, finds `src/geekbot` on its own, and its editable installs use the plain `.pth` file mentioned above. That matters because static tools (mypy, VS Code's Pylance) can follow a `.pth` file. They can't follow the import hooks some other editable modes use. setuptools is the classic alternative and would work too.

**Dependencies vs extras**
- `dependencies` = what Geekbot needs to **run**.
- `[project.optional-dependencies]` = named groups ("extras") you opt into. `dev` holds the developer tools (pytest, ruff, mypy, pre-commit, pip-tools). Later, `eval` will hold the benchmark tools the app doesn't need.
- `pip install -e ".[dev]"` installs the package plus the `dev` extra.

**Our rule:** a dependency enters `pyproject.toml` in the step whose code first imports it. S0.1 has **zero** runtime dependencies. That keeps every addition small, explained and reviewed.

**Version: `2.0.0.dev0`.** The format follows PEP 440: `.dev0` means "in development, before 2.0.0". The code doesn't repeat the number. `geekbot.__version__` reads it from the installed metadata (`importlib.metadata.version("geekbot")`), so `pyproject.toml` stays the single source of truth.

**`requires-python = ">=3.12"`, no upper cap.** A cap like `<3.13` makes resolvers fall back to old releases or fail outright, and it ages badly. Instead, CI tests 3.13 (S0.5).

**License: `license = "MIT"`.** PEP 639 lets you write the license as a standard SPDX identifier (a short official code such as `MIT` or `Apache-2.0`) instead of free text.

---

## 3. Entry points: how `geekbot` becomes a command
```toml
[project.scripts]
geekbot = "geekbot.cli:main"
```
At install time, pip writes a small launcher, `geekbot.exe`, into the environment's `Scripts\` folder. The launcher imports `geekbot.cli` and runs `sys.exit(main())`.

That explains two things:
- Once the env is active, `geekbot --version` works from **any** folder.
- Changing this table needs a re-install, because the launcher is generated at install time.

`main()` returns an **exit code**: 0 means success, anything else means failure. Scripts and CI rely on exit codes to know if a command worked.

S0.1's `cli.py` is a ~15-line placeholder (just `--version`). S0.2 replaces it with a real CLI built on Typer.

---

## 4. conda vs pip

| | conda | pip |
|---|---|---|
| Manages | a whole environment, **including non-Python programs**: the Python interpreter itself, C libraries, CUDA | Python packages only |
| Gets packages from | channels (conda-forge, Anaconda's `defaults`) | PyPI |
| Solves | every package in the env together | the Python packages you ask for |
| Knows about the other? | only partly sees pip installs | not at all |

**The classic pitfall: mixing.** Install `click` with conda, then again with pip, and two installers think they own the same files. Upgrades and uninstalls then break in confusing ways. The usual advice is "conda first, pip last, and never go back to conda".

**Our pattern avoids mixing entirely:**
- conda installs only **Python 3.12 + pip**, from conda-forge alone (`nodefaults` keeps Anaconda's own channel out, so every conda package comes from one place).
- **Every** Python package, even pip-tools, comes from pip, pinned by the lock.
- One owner per package; nothing is installed twice.

**Why conda at all, then?** It provides an isolated Python 3.12 without touching your system Python 3.13.9 (D-010, your choice). A plain venv could do the same job; conda was chosen.

---

## 5. Lock files: ranges vs pins

| File | Says | Example |
|---|---|---|
| `pyproject.toml` | what **should** work (ranges) | `ruff>=0.16` |
| `requirements/lock-win.txt` | what **was** tested (exact pins, *including every dependency of a dependency*) | `ruff==0.16.10` |

Ranges keep the package flexible. Pins make the environment reproducible: re-create it next month and you get the same versions, not whatever was released in between.

**pip-compile** (from pip-tools) turns the first into the second:
```
pip-compile --extra dev --strip-extras --output-file requirements/lock-win.txt pyproject.toml
```
- `--extra dev`: include the `dev` extra.
- `--strip-extras`: write `pkg==1.2`, not `pkg[extra]==1.2`. Cleaner, and it will become the default in pip-tools 8.
- **Re-running keeps existing pins** and only adds or changes what's needed. `--upgrade` (everything) or `--upgrade-package ruff` (one) bump versions **on purpose**, so every upgrade is a visible diff in a commit.

**Why the lock is Windows-only.** Dependencies can carry *markers*, e.g. click needs `colorama; platform_system == "Windows"`. pip-compile evaluates markers for the machine it runs on, so a Windows lock lists colorama and a Linux lock wouldn't. Hence the name `lock-win.txt`.

Linux CI instead resolves fresh from the pyproject ranges (S0.5). That also gives early warning when a new upstream release breaks us. Tools that write one cross-platform lock exist (the new PEP 751 `pylock.toml` format, uv, Poetry), but we only have one dev box.

**What the lock deliberately leaves out**
- **pip and setuptools.** pip-tools calls them "unsafe" to pin and skips them unless you pass `--allow-unsafe`. That suits us: pip belongs to conda here.
- **Hashes** (`--generate-hashes`). Hashes let pip refuse a package whose file was tampered with on PyPI. But pip's hash-checking mode refuses editable installs (`-e .`): a source folder has no single file to hash. Our env file installs the lock and `-e .` in one pip call. Skipping hashes is a reasonable trade-off for a local dev box; a deployed build should use them.

**Workflow when a later step adds a dependency**
1. I add it to `pyproject.toml`.
2. You run pip-compile; the lock gains the new pins.
3. You update the env: `conda env update -f environment.yml --prune`.
4. `pyproject.toml` and the lock are committed **together**.

Note: `--prune` removes conda packages that are no longer listed, but it does **not** uninstall pip packages dropped from the lock. To get an exact copy, re-create the env (a few minutes).

---

## 6. `environment.yml`
```yaml
name: geekbot
channels:
  - conda-forge
  - nodefaults
dependencies:
  - python=3.12
  - pip
  - pip:
      - -r requirements/lock-win.txt
      - -e .
```
conda installs Python + pip, then hands the `pip:` list to pip. The paths are relative to the yml file.

**Chicken and egg:** this file needs the lock, and the lock can only be generated inside an environment. So S0.1 bootstraps once with plain commands (create a bare env, install `-e ".[dev]"`, run pip-compile), then **deletes the env and rebuilds it from this file**. The rebuild is the real test: the environment can be reproduced from the repo alone.

---

## 7. The quality tools

### ruff: linter + formatter
Written in Rust and very fast. It replaces flake8, isort, black and pyupgrade.
- `ruff check .` finds problems; `ruff check --fix .` fixes the safe ones.
- `ruff format .` rewrites the layout (spacing, quotes, line breaks), so style is never a review topic.
- **Since 0.16, ruff enables ~413 rules by default** (bugbear, pyupgrade, import sorting, much of pylint…). We start from those curated defaults instead of hand-picking. That's why the floor is `ruff>=0.16`.
- **Excluded:**
  - the v1 files, untouched until step S4.4 (the Streamlit UI step) removes them
  - Markdown files, because ruff 0.16 also reformats Python code blocks inside `.md` files, and our notes sometimes quote v1 code exactly as it was
- **Magic numbers in tests are allowed** (`PLR2004`): `assert len(x) == 3` is clearer than a named constant.

### mypy --strict: static type checker
It reads the type hints and proves, **without running the code**, that you never pass e.g. a `str | None` where a `str` is required.
- `strict = true` switches on a bundle of checks: every function annotated, no silent `Any` leaking through, no unused `# type: ignore`.
- It catches a class of bugs tests miss: the branch no test happened to run.
- mypy 2.x also turned two stricter checks on by default (`--strict-bytes`, `--local-partial-types`).

### pytest: test runner
Our settings:
- `--strict-markers`: a typo in a marker name is an error, not a test that silently never runs.
- `--strict-config`: a typo in pytest's config is an error too.
- `--import-mode=importlib`: pytest's recommendation for new projects; test folders don't need `__init__.py` files.
- **`-m "not live"` by default.** Tests that need Ollama or the GPU are marked `@pytest.mark.live` and skipped unless you ask. A plain `pytest` never touches the GPU your CSV project uses. `pytest -m live` runs them deliberately (the last `-m` on the command line wins). pytest reports the skipped ones as "deselected".

### pre-commit: checks at every commit
pre-commit runs small checks ("hooks") on the files you staged, every time you `git commit` (also from VS Code's Source Control panel).
- If a hook **fixes** a file (e.g. ruff reformats it), the commit stops. You look at the change, stage it, and commit again.
- Hooks run in small isolated environments that pre-commit keeps in its own cache, separate from the `geekbot` env. So the ruff hook's `rev` must match the ruff version in the lock, or the hook and `ruff check .` could disagree.
- **Pitfall:** pre-commit passes file names to ruff explicitly, and ruff ignores its own exclude list for explicitly named files unless `force-exclude = true`. Without that setting, the hook would lint the v1 files.
- **mypy and pytest are not hooks.** Inside pre-commit's isolated env, mypy can't see our dependencies, so it checks badly. Tests are too slow to run on every commit. Both run manually now, and in CI from S0.5.

---

## 8. How this maps to Geekbot

| File | Role |
|---|---|
| `environment.yml` | conda: Python 3.12 + pip from conda-forge; pip: the lock + the editable project |
| `pyproject.toml` | metadata, the `dev` extra, the `geekbot` command, ruff/mypy/pytest config |
| `requirements/lock-win.txt` | exact versions, generated by you with pip-compile; never edited by hand |
| `src/geekbot/__init__.py` | package root; `__version__` read from metadata |
| `src/geekbot/py.typed` | empty marker: "this package ships type hints" (PEP 561), so code importing geekbot (e.g. your CSV project at Gate C) gets type-checked against it |
| `src/geekbot/cli.py` | placeholder `geekbot --version`; replaced in S0.2 |
| `tests/unit/test_smoke.py` | proves the install, the version wiring and the command work |
| `.pre-commit-config.yaml` | hygiene hooks + ruff at commit time |
| `.gitignore` | keeps envs, caches, `data/` and `.env` out of git |
| `LICENSE` | MIT |

---

## Interview questions

**1. Why use a src layout?**
In a flat layout the package folder sits on `sys.path` when you run from the repo root, so tests can pass against an uninstalled or wrongly built package. With src, the package is importable only once installed, so tests run against what users actually get. The cost is one editable install.

**2. Why keep ranges in `pyproject.toml` and exact pins in a separate lock, instead of pinning in pyproject?**
They answer different questions. Ranges say what the code supports and keep it resolvable alongside other packages. The lock records the exact tested set, including transitive dependencies, so the environment can be rebuilt identically. Pins inside pyproject would also block CI's fresh resolution from finding upstream breakage early.

**3. Your lock is Windows-only. How do you know Linux works?**
pip-compile evaluates environment markers (like `platform_system == "Windows"`) for the machine it runs on, so the lock is platform-specific. Linux CI therefore installs fresh from the pyproject ranges on 3.12 and 3.13. If that run breaks while the Windows-lock run passes, an upstream release changed something.

**4. Why does conda install only Python and pip in this project?**
conda and pip don't fully see each other's installs. A package installed by both gets two owners, and upgrades or uninstalls corrupt it. Letting conda provide only the interpreter, and pip every Python package, gives each package exactly one owner, pinned by one lock.

**5. Your pre-commit ruff hook flags a file you excluded in pyproject. Why?**
pre-commit passes the staged file names explicitly, and ruff applies `exclude` only to files it discovers itself, not to files it's handed by name. `force-exclude = true` makes ruff honour the exclude list in both cases.

**6. Why are there no hashes in the lock?**
Hash-checking mode makes pip refuse any requirement it can't hash, and an editable install of a source folder has no single file to hash. We install the lock and `-e .` together, so hashes would break the env. The trade-off (no tamper check on downloads) is acceptable for a local dev box; a deployed build should install a non-editable wheel with hashes.
