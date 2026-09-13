import re

with open("Quant16 Linear Regression Kernel Smoothing and Interview Classics.md", "r") as f:
    lines = f.readlines()

# find the start of the english part
start_idx = 0
for i, line in enumerate(lines):
    if line.startswith("# Quant 11 · Linear regression and kernel smoothing"):
        start_idx = i
        break

en_lines = lines[start_idx:]

content = "".join(en_lines)

# Fix the references: Combine Zhou and Crack into one line.
# Original:
# - Zhou, *A Practical Guide to Quantitative Finance Interviews*
# 
# - Crack, *Heard on the Street*
# 
content = re.sub(r"- Zhou, \*A Practical Guide to Quantitative Finance Interviews\*\n\n- Crack, \*Heard on the Street\*",
                 r"- Zhou, *A Practical Guide to Quantitative Finance Interviews*; Crack, *Heard on the Street*", content)

with open("Quant16 Linear Regression Kernel Smoothing and Interview Classics.en.md", "w") as f:
    f.write(content)
