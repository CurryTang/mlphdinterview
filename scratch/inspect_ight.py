import re

with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "ight" in l:
        print(f"Line {i+1}: {repr(l)}")
