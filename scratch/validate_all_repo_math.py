import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)
err_count = 0

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    display_math_open = False
    in_code_block = False
    
    for idx, l in enumerate(lines):
        if l.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
            
        dollar2 = l.count("$$")
        if dollar2 % 2 != 0:
            display_math_open = not display_math_open
            
        if not display_math_open and dollar2 == 0:
            cleaned = l.replace(r"\$", "")
            cleaned = re.sub(r'`[^`]*`', '', cleaned)
            single_dollars = cleaned.count("$")
            if single_dollars % 2 != 0:
                print(f"WARN {fpath}:{idx+1}: Unbalanced $ -> {l.strip()}")
                err_count += 1
                
    if display_math_open:
        print(f"ERROR {fpath}: Unclosed $$ at EOF")
        err_count += 1

print(f"Total math syntax warnings/errors across repo: {err_count}")
