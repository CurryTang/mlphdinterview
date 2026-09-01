import glob

files = glob.glob("notes/MLCoding/*.md")
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    issues = []
    if "\x0c" in content:
        issues.append("formfeed \\x0c")
    if "\r" in content:
        issues.append("carriage return \\r")
    if "	ext" in content:
        issues.append("tab text")
    if "	o" in content:
        issues.append("tab to")
    if " pprox" in content:
        issues.append("pprox")
    if "ight)" in content:
        issues.append("ight)")
    if issues:
        print(f"Issues in {fpath}: {issues}")
print("Verification complete!")
