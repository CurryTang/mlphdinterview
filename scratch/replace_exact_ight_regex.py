import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    # Replace all occurrences of " ight)", "  ight)", "\tight)" with " \right)"
    content = re.sub(r'\s+ight\)', r' \\right)', content)
    content = re.sub(r'\s+ight\]', r' \\right]', content)
    content = re.sub(r'\s+ight\\\}', r' \\right\\}', content)
    content = re.sub(r'\(ight\)', r'(\\right)', content)
    content = content.replace(" ight)", " \\right)")

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed: {fpath}")

print("Done regex replacement of ight)!")
