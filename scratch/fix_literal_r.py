import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    # Remove literal "\r" or carriage returns before \right
    content = content.replace(" \\r\\right)", " \\right)")
    content = content.replace("\\r\\right)", "\\right)")
    content = content.replace(" \\r \\right)", " \\right)")
    content = content.replace(" \\r\\right]", " \\right]")
    content = content.replace("\\r\\right]", "\\right]")
    content = content.replace(" \\r \\right]", " \\right]")
    content = content.replace(" \\r\\right\\}", " \\right\\}")
    content = content.replace("\\r\\right\\}", "\\right\\}")
    content = content.replace(" \\r", " ")
    content = content.replace("\\r", "")

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Cleaned literal \\r from {fpath}")

print("Cleaned literal \\r from all files!")
