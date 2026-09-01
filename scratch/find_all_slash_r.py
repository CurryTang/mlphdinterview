import glob

files = glob.glob("notes/**/*.md", recursive=True)
count = 0
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for literal "\r" in the text
    if "\\r" in content:
        print(f"FOUND literal \\r in {fpath}")
        # find line numbers
        lines = content.split("\n")
        for idx, l in enumerate(lines):
            if "\\r" in l:
                print(f"  Line {idx+1}: {l.strip()}")
                count += 1

print(f"Total literal \\r found: {count}")
