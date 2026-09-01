import glob
import re

files = glob.glob("notes/**/*.md", recursive=True)

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    
    # 1. Fix tab-corrupted commands
    content = content.replace("	imes", "\\times")
    content = content.replace("	ext", "\\text")
    content = content.replace("	o", "\\to")
    content = content.replace("	heta", "\\theta")
    content = content.replace("	au", "\\tau")
    content = content.replace("	au", "\\tau")
    content = content.replace(" pprox", " \\approx")
    content = content.replace("\x07pprox", "\\approx")
    content = content.replace("\x0crac", "\\frac")
    content = content.replace("\x0c", "\\f")
    content = content.replace("\r", "")
    
    # 2. Fix broken ight in all variations
    content = re.sub(r'\n\s*ight\)', r' \\right)', content)
    content = re.sub(r'\\left\(\s*([^()\n]+?)\s*\n\s*ight\)', r'\\left( \1 \\right)', content)
    content = re.sub(r'\\left\(\s*([^()]+?)\s*ight\)', r'\\left( \1 \\right)', content)
    content = re.sub(r'\\left\[\s*([^\[\]]+?)\s*ight\]', r'\\left[ \1 \\right]', content)
    content = re.sub(r'\\left\\\{\s*([^{}]+?)\s*ight\\\}', r'\\left\\{ \1 \\right\\}', content)
    content = content.replace(" ight)", " \\right)")
    content = content.replace(" ight]", " \\right]")
    content = content.replace(" ight\\}", " \\right\\}")
    content = content.replace(" ight.", " \\right.")
    content = content.replace("\tight)", "\\right)")
    content = content.replace("ight)", "\\right)")

    # 3. Specifically fix Plackett-Luce formula
    pl_old = r"""$$\mathcal{L}_{\text{Plackett-Luce}} = -\sum_{k=1}^K \log \left( \frac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} 
\right)$$"""
    pl_new = r"""$$\mathcal{L}_{\text{Plackett-Luce}} = -\sum_{k=1}^K \log \left( \frac{\exp(s_{\pi_k})}{\sum_{j=k}^K \exp(s_{\pi_j})} \right)$$"""
    content = content.replace(pl_old, pl_new)

    if content != orig:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed: {fpath}")

print("Comprehensive LaTeX sweep complete!")
