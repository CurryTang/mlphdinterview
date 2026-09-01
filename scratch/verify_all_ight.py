import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)
count = 0
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for idx, l in enumerate(lines):
        # Look for ight| or \left ... ight
        if "ight|" in l:
            print(f"STILL FOUND ight|: {fpath}:{idx+1} -> {l.strip()}")
            count += 1
        if "ight)" in l and "right)" not in l and "height)" not in l and "weight)" not in l:
            print(f"STILL FOUND ight): {fpath}:{idx+1} -> {l.strip()}")
            count += 1

print(f"Total remaining errors: {count}")
