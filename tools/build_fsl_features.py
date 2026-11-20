import ast, os, re, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPLY = "--apply" in sys.argv

FUNCS = [  # smooth_landmarks must precede resolve_and_smooth_hands
    "get_detailed_face_and_body_anchors", "extract_hand_frame", "get_raw_hand_anchors",
    "build_motion_dynamics_and_trajectory", "extract_face_frame",
    "normalize_hand_sequence", "smooth_landmarks", "resolve_and_smooth_hands",
]
TARGETS = ["main.py", "Words_Learning/collect_word_data.py", "Words_Learning/train_word_model.py"]

HEADER = '''"""Shared FSL feature code. Single source for main.py, collect_word_data.py, train_word_model.py."""
import math

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Word data was recorded unsmoothed. Flip only after re-recording and retraining.
SMOOTH_LANDMARKS = False

'''


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def top_funcs(src):
    return {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


main_src = read("main.py")
main_nodes = top_funcs(main_src)
missing = [f for f in FUNCS if f not in main_nodes]
if missing:
    sys.exit(f"missing in main.py: {missing}")

body = "\n\n\n".join(ast.get_source_segment(main_src, main_nodes[f]) for f in FUNCS)
out = os.path.join(ROOT, "fsl_features.py")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(HEADER + body + "\n")
print(f"wrote {out}")

if not APPLY:
    sys.exit("dry run: re-run with --apply to rewrite the 3 files")

for rel in TARGETS:
    src = read(rel)
    nodes = top_funcs(src)
    names = [f for f in FUNCS if f in nodes]
    if not names:
        continue
    lines = src.splitlines(keepends=True)
    ranges = sorted((nodes[n].lineno - 1, nodes[n].end_lineno) for n in names)
    in_root = "/" not in rel
    imp = "from fsl_features import (\n" + "".join(f"    {n},\n" for n in names) + ")\n"
    if not in_root:
        imp = "import sys as _sys\n_sys.path.insert(0, ROOT_DIR)\n" + imp
    new, i, inserted = [], 0, False
    while i < len(lines):
        hit = next((r for r in ranges if r[0] == i), None)
        if hit:
            if not inserted:
                new.append(imp + "\n")
                inserted = True
            i = hit[1]
            continue
        if rel == "main.py" and re.match(r"SMOOTH_LANDMARKS\s*=", lines[i]):
            i += 1
            continue
        new.append(lines[i])
        i += 1
    shutil.copy2(os.path.join(ROOT, rel), os.path.join(ROOT, rel + ".bak"))
    with open(os.path.join(ROOT, rel), "w", encoding="utf-8", newline="") as f:
        f.write("".join(new))
    print(f"{rel}: removed {len(names)} funcs, backup {rel}.bak")
