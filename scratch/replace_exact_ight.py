import glob

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    content = content.replace(" ight)", " \\right)")
    content = content.replace(" ight]", " \\right]")
    content = content.replace(" ight\\}", " \\right\\}")
    content = content.replace(" ight.", " \\right.")
    content = content.replace("( ight", "(\\right")
    content = content.replace("	ext", "\\text")
    content = content.replace("	o", "\\to")

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {fpath}")

print("Exact ight replace done!")
