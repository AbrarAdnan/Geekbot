# Lock files

`lock-win.txt` pins the exact version of every Python package in the `geekbot` env on Windows, including the dependencies of dependencies. **It is generated; never edit it by hand.** Why it exists and why it's Windows-only: [lesson 01, §5](../project/learn/01-packaging.md).

**Regenerate it** whenever the dependencies in `pyproject.toml` change. Run it in the `geekbot` env, from the repo root:
```
pip-compile --extra dev --strip-extras --output-file requirements/lock-win.txt pyproject.toml
```
- Add one `--extra <name>` per extra defined in `pyproject.toml` (e.g. `--extra eval` once it exists).
- Existing pins are kept. To upgrade on purpose, add `--upgrade-package <name>` (one package) or `--upgrade` (everything).

**Then:**
1. Update the env: `conda env update -f environment.yml --prune`.
2. Commit `pyproject.toml` and the lock together.
