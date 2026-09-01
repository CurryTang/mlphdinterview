import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    orig = content

    # Replace ight) with \right)
    content = content.replace(" ight)", " \\right)")
    content = content.replace(" ight]", " \\right]")
    content = content.replace(" ight|", " \\right|")
    content = content.replace(" ight\\}", " \\right\\}")
    content = content.replace(" ight.", " \\right.")
    content = re.sub(r'([^\\])right\)', r'\1\\right)', content)
    content = re.sub(r'([^\\])right\]', r'\1\\right]', content)
    content = re.sub(r'([^\\])right\|', r'\1\\right|', content)

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Repaired right parens: {fpath}")

print("Exact right parens fix complete!")
