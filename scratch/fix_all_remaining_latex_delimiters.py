import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    orig = content
    
    # 1. Fix ceil / floor / abs delimiters
    content = content.replace(" ight ceil", " \\right\\rceil")
    content = content.replace(" ight floor", " \\right\\rfloor")
    content = content.replace(" ight|", " \\right|")
    content = content.replace("ight|", "\\right|")
    content = content.replace(" ight)", " \\right)")
    content = content.replace("ight)", "\\right)")
    content = content.replace(" ight]", " \\right]")
    content = content.replace("ight]", "\\right]")
    content = content.replace(" ight\\}", " \\right\\}")
    content = content.replace("ight\\}", "\\right\\}")
    content = content.replace(" ight.", " \\right.")
    content = content.replace("ight.", "\\right.")
    
    # Also clean \left\lfloor and \right\rfloor
    content = re.sub(r'\\left\\lfloor\s*(.*?)\s*ight\s*ceil', r'\\left\\lfloor \1 \\right\\rceil', content)

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Repaired: {fpath}")

print("Delimiters sweep complete!")
