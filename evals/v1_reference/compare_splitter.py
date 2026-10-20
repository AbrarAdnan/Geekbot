"""Step 2 of the splitter check (CP2): cut the same texts with v1's real splitter and compare.

Run this in the ``geekbot-v1`` environment (``langchain-text-splitters`` 0.3.11), from the root of
the v1 worktree, on the file ``dump_splitter_cases.py`` wrote. v1's own ``TEXT_SPLITTER`` (from
``storage_utils``) cuts every text; the chunks must equal, string for string, the ones rung R0 of
the ablation made with the 1.x splitter. Any difference is printed with its text id.

    python <path>/compare_splitter.py <path>/scratch/splitter-cases.json
"""

import json
import sys
from pathlib import Path


def main(path: Path) -> int:
    sys.path.insert(0, str(Path.cwd()))  # the v1 worktree
    from langchain.schema import Document
    from storage_utils import TEXT_SPLITTER

    cases = json.loads(path.read_text(encoding="utf-8"))
    different = []
    total_chunks = 0
    for case in cases:
        v1_chunks = [
            d.page_content
            for d in TEXT_SPLITTER.split_documents([Document(page_content=case["text"])])
        ]
        total_chunks += len(v1_chunks)
        if v1_chunks != case["chunks"]:
            different.append(case["id"])
    print(f"{len(cases)} texts, {total_chunks} chunks cut by v1's splitter")
    print(f"identical for {len(cases) - len(different)} of {len(cases)} texts")
    for case_id in different[:20]:
        print(f"  DIFFERENT: {case_id}")
    return 0 if not different else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
