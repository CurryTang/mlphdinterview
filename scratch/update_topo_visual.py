# -*- coding: utf-8 -*-

with open("src/App.jsx", "r", encoding="utf-8") as f:
    app_jsx = f.read()

# 1. Read the new component code from create_topo_component.py
with open("scratch/create_topo_component.py", "r", encoding="utf-8") as f:
    topo_py = f.read()

start_mark = 'code_topo = r"""\n'
end_mark = '"""\n\nprint("Code generated successfully!")'
new_component_code = topo_py[topo_py.find(start_mark) + len(start_mark) : topo_py.rfind(end_mark)]

# 2. Replace ForeignDictionaryTopoVisual in App.jsx
old_func_start = app_jsx.find("function ForeignDictionaryTopoVisual() {")
old_func_end = app_jsx.find("function CheapestFlightsBellmanVisual() {")

if old_func_start != -1 and old_func_end != -1:
    app_jsx = app_jsx[:old_func_start] + new_component_code + "\n" + app_jsx[old_func_end:]
    # Replace render call in MarkdownPre
    app_jsx = app_jsx.replace("<ForeignDictionaryTopoVisual />", "<TopologicalSortVisual />")
    with open("src/App.jsx", "w", encoding="utf-8") as f:
        f.write(app_jsx)
    print("Successfully replaced ForeignDictionaryTopoVisual in App.jsx with TopologicalSortVisual!")
else:
    print("Error finding function boundaries in App.jsx")

