import glob

files = [
    "notes/quant/Quant11 Martingales Stopping Times Random Walks.en.md",
    "notes/quant/Quant11 Martingales Stopping Times Random Walks.md",
    "notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md",
    "notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md"
]

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.replace(" \\r\\right|", " \\right|")
    content = content.replace("\\r\\right|", "\\right|")
    content = content.replace("  \\r\\right|", " \\right|")
    content = content.replace(" \\r \\right|", " \\right|")
    
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed final 4: {fpath}")

