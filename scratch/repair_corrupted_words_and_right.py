import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    orig = content

    # 1. Fix corrupted English words where "ight" was erroneously replaced by "\right"
    content = content.replace("we\\right", "weight")
    content = content.replace("We\\right", "Weight")
    content = content.replace("he\\right", "height")
    content = content.replace("He\\right", "Height")
    content = content.replace("Spotl\\right", "Spotlight")
    content = content.replace("spotl\\right", "spotlight")
    content = content.replace("br\\right", "bright")
    content = content.replace("Br\\right", "Bright")
    content = content.replace("sl\\right", "slight")
    content = content.replace("Sl\\right", "Slight")
    content = content.replace("fl\\right", "flight")
    content = content.replace("Fl\\right", "Flight")
    content = content.replace("l\\right", "light")
    content = content.replace("L\\right", "Light")
    content = content.replace("r\\right", "right")
    content = content.replace("R\\right", "Right")
    content = content.replace("n\\right", "night")
    content = content.replace("N\\right", "Night")

    # 2. Fix \r\right in LaTeX
    content = content.replace("\\r\\right)", "\\right)")
    content = content.replace("\\r\\right]", "\\right]")
    content = content.replace("\\r\\right\\}", "\\right\\}")
    content = content.replace("\\r\\right|", "\\right|")
    content = content.replace("\\r\\right.", "\\right.")
    content = content.replace("\\r\\right", "\\right")
    content = content.replace(" \\r", " ")
    content = content.replace("\\r", "")

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Repaired: {fpath}")

print("Word and delimiter repair complete!")
