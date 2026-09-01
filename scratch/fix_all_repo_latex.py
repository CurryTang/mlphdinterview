import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    # Replace escaped control chars
    content = content.replace("\x0crac", "\\frac")
    content = content.replace("\x0c", "\\f")
    content = content.replace("\x07pprox", "\\approx")
    content = content.replace(" pprox", " \\approx")
    content = content.replace("\r", "")
    content = content.replace("	ext", "\\text")
    content = content.replace("	o", "\\to")
    
    # Fix broken \left( ... ight)
    content = re.sub(r'\\left\(\s*([^()]+?)\s*ight\)', r'\\left( \1 \\right)', content)
    content = re.sub(r'\\left\[\s*([^\[\]]+?)\s*ight\]', r'\\left[ \1 \\right]', content)
    content = re.sub(r'\\left\\\{\s*([^{}]+?)\s*ight\\\}', r'\\left\\{ \1 \\right\\}', content)
    content = content.replace(" ight)", " \\right)")
    content = content.replace(" ight]", " \\right]")
    content = content.replace(" ight\\}", " \\right\\}")
    content = content.replace(" ight.", " \\right.")
    content = content.replace("\tight)", "\\right)")

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {fpath}")

print("All repository LaTeX fixed!")
