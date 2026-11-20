import ast
import difflib
import os


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [
    "main.py",
    "Words_Learning/collect_word_data.py",
    "Words_Learning/train_word_model.py",
]
FUNCS = [
    "get_detailed_face_and_body_anchors",
    "extract_hand_frame",
    "get_raw_hand_anchors",
    "build_motion_dynamics_and_trajectory",
    "extract_face_frame",
    "normalize_hand_sequence",
    "smooth_landmarks",
    "resolve_and_smooth_hands",
]


def load(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}


srcs = {p: load(p) for p in FILES}
for fn in FUNCS:
    have = {p: s[fn] for p, s in srcs.items() if fn in s}
    if len(have) < 2:
        print(f"{fn}: only in {list(have)}")
        continue
    base_p, base = next(iter(have.items()))
    for p, s in have.items():
        if s == base:
            continue
        print(f"{fn}: DIFFERS {base_p} vs {p}")
        print("\n".join(difflib.unified_diff(base.splitlines(), s.splitlines(), lineterm="", n=1)))
    if all(s == base for s in have.values()):
        print(f"{fn}: identical in {list(have)}")
