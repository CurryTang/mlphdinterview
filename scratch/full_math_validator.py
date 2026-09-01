import re

with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.split("\n")

# Check $$ blocks pairing
display_math_open = False
for idx, l in enumerate(lines):
    # Count $$
    dollar2 = l.count("$$")
    if dollar2 == 1:
        display_math_open = not display_math_open
    elif dollar2 == 2:
        pass # open and closed on same line: $$...$$
    elif dollar2 > 2:
        print(f"Line {idx+1}: More than 2 $$ in line: {l}")
    
    # Check for unmatched $
    # Remove code blocks and display math
    if not display_math_open and dollar2 == 0:
        # Check single dollar balance
        # remove escaped \$
        cleaned = l.replace(r"\$", "")
        # remove inline code `...`
        cleaned = re.sub(r'`[^`]*`', '', cleaned)
        single_dollars = cleaned.count("$")
        if single_dollars % 2 != 0:
            print(f"Line {idx+1}: Unbalanced single dollar ($): {l}")

print(f"Display math block open state at EOF: {display_math_open}")
print("Validation complete!")
