import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, l in enumerate(lines):
        # Find ight that is not preceded by 'r' or 'w' (e.g. weight, right, light, night, etc.)
        matches = re.findall(r'(?<![rwlnsfbehmptW])ight[^\w\s]', l)
        # Also check for ight| or ight) or ight]
        for m in re.finditer(r'(\\left[^\n]+?)(ight[\)|\]\.\\\}])', l):
            print(f"{fpath}:{idx+1} -> {m.group(0)}")
        if "ight|" in l:
            print(f"{fpath}:{idx+1} ight| -> {l.strip()}")
        if "ight)" in l and "right)" not in l:
            print(f"{fpath}:{idx+1} ight) -> {l.strip()}")
        if "ight]" in l and "right]" not in l:
            print(f"{fpath}:{idx+1} ight] -> {l.strip()}")
        if "ight\\}" in l and "right\\}" not in l:
            print(f"{fpath}:{idx+1} ight}} -> {l.strip()}")
