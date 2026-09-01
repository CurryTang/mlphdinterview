import re

# 1. Fix MLCoding00B.md
with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md", "r", encoding="utf-8") as f:
    content = f.read()

# Replace corrupted escape sequences
content = content.replace("\x0crac", "\\frac")
content = content.replace("\x07pprox", "\\approx")
content = content.replace(" pprox", "\\approx")
content = content.replace("\right)", "\\right)")
content = content.replace("\r", "")
content = content.replace("	ext", "\\text")
content = content.replace("	o", "\\to")

# Let's clean any remaining corrupted LaTeX patterns
content = re.sub(r'\\left\(\s*\\frac\{1\}\{V\}\s*ight\)', r'\\left( \\frac{1}{V} \\right)', content)
content = re.sub(r'\\left\(\s*([^()]+)\s*ight\)', r'\\left( \1 \\right)', content)
content = content.replace("ight)", "\\right)")

with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.md", "w", encoding="utf-8") as f:
    f.write(content)

# 2. Fix MLCoding00B.en.md
with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md", "r", encoding="utf-8") as f:
    content_en = f.read()

content_en = content_en.replace("\x0crac", "\\frac")
content_en = content_en.replace("\x07pprox", "\\approx")
content_en = content_en.replace(" pprox", "\\approx")
content_en = content_en.replace("\right)", "\\right)")
content_en = content_en.replace("\r", "")
content_en = content_en.replace("	ext", "\\text")
content_en = content_en.replace("	o", "\\to")
content_en = re.sub(r'\\left\(\s*\\frac\{1\}\{V\}\s*ight\)', r'\\left( \\frac{1}{V} \\right)', content_en)
content_en = re.sub(r'\\left\(\s*([^()]+)\s*ight\)', r'\\left( \1 \\right)', content_en)
content_en = content_en.replace("ight)", "\\right)")

with open("notes/MLCoding/MLCoding00B LLM Basics Decoder Only Precision Alignment Distillation.en.md", "w", encoding="utf-8") as f:
    f.write(content_en)

print("Cleaned MLCoding00B LaTeX formulas")
