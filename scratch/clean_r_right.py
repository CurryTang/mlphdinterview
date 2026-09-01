import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    # Clean any leftover \r before \right or \left
    content = re.sub(r'\\r+\s*\\right', r'\\right', content)
    content = re.sub(r'\\r+\s*\\left', r'\\left', content)
    content = re.sub(r'\\r+\s*', r' ', content)
    content = re.sub(r'\\right\s*\\right', r'\\right', content)

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Cleaned {fpath}")

print("Done cleaning \\r patterns!")
