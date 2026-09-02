# -*- coding: utf-8 -*-

with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

target_start_en = "## Module 8: Eulerian Path, Reconstruct Itinerary"
pos_start_en = en.find(target_start_en)
pos_code_en = en.find("```python\nimport heapq\nfrom collections import defaultdict\nfrom typing import List\n\n\nclass Solution:\n    def findItinerary", pos_start_en)

euler_replacement_en = r"""---

## Module 8: Eulerian Paths (Hierholzer's Algorithm & Reconstruct Itinerary)

### 16. Reconstruct Itinerary (LeetCode 332)

#### Detailed Problem Description
You are given a list of airline tickets `tickets` where `tickets[i] = [from_i, to_i]` represents the departure and arrival airports of a one-way flight. Reconstruct the complete flight itinerary in order and return it.

**Core Rules & Invariants**:
1. **Mandatory Departure**: All itineraries must begin from Kennedy International Airport `"JFK"`;
2. **Every Ticket Used Exactly Once (Full Edge Coverage)**: You must use all tickets **exactly once** (if duplicate tickets exist, each ticket represents a distinct directed edge and must be flown separately);
3. **Nodes Can Be Re-visited**: Airports can be visited multiple times (in-degrees and out-degrees can be $> 1$);
4. **Lexicographically Smallest Order**: If there are multiple valid itineraries, return the itinerary that has the **smallest lexical order** when read as a single string (e.g., `["JFK", "ATL", "JFK"]` has a smaller lexical order than `["JFK", "SFO", "JFK"]`);
5. **Guarantee of Solution**: You may assume all inputs have at least one valid itinerary.

```text
Eulerian Path & Dead-End Traps (Example: tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]])
                ┌────────────────┐
                │                ▼
             [JFK] (Start) ⇄ [NRT] (Sub-circuit)
                │
                ▼ (Dead-end branch: out-degree = 0)
              [KUL] (Destination)

★ Greedy Trap: JFK has destinations 'KUL' and 'NRT'. If we greedily take smaller 'KUL' in preorder, we get trapped in KUL (no outgoing edges), leaving the NRT circuit stranded!
★ Hierholzer Resolution: Post-order recording. KUL hits the dead end first and gets pushed to route first. Backtracking traverses the remaining NRT circuit and pushes JFK last. Reversing gives the correct path: ["JFK", "NRT", "JFK", "KUL"]!
```

#### Graph Theoretical Formulation: Eulerian Path vs. Hamiltonian Path

Most graph problems (e.g. topological sort, number of islands, course schedule) focus on visiting each **vertex** once (vertex coverage). Reconstruct Itinerary requires using every **ticket (edge)** once, which is an **Eulerian Path** problem:

| Dimension | Eulerian Path / Circuit | Hamiltonian Path / Cycle |
|---|---|---|
| **Core Definition** | Traverses **every edge exactly once** (vertices can be revisited) | Traverses **every vertex exactly once** (edges may be skipped) |
| **Existence Condition** | **Directed Graph**: At most one vertex with $\text{out} - \text{in} = 1$ (start), at most one with $\text{in} - \text{out} = 1$ (end), all others $\text{out} == \text{in}$ | **NP-Hard** (no polynomial necessary & sufficient condition) |
| **Classic Problems** | Seven Bridges of Königsberg, Line Tracing, Ticket Itinerary (LeetCode 332) | Travelling Salesperson Problem (TSP), Knight's Tour |
| **Algorithm & Complexity** | **Hierholzer's Algorithm: $\mathcal{O}(E \log E)$** (polynomial time) | Exponential backtracking / DP: $\mathcal{O}(2^V \cdot V^2)$ (NP-Hard) |

#### Hierholzer's Algorithm Mechanics: Post-order DFS & Sub-circuit Splicing

1. **Adjacency Representation**: Store destinations for each airport in a **Min-Heap**, ensuring greedy selection of lexicographically smallest candidate destinations;
2. **In-place Edge Removal**: Each DFS step pops the smallest edge from the heap (consuming the ticket);
3. **Post-order Appending**:
   - Append an airport to `route` only when its heap is exhausted (no more tickets left from this airport);
   - Dead-end destination airports (such as `KUL`) exhaust their tickets first and get appended first;
   - As recursion unwinds, outer airports traverse any remaining sub-circuits before appending themselves;
4. **Final Reverse**: Reversing the post-order sequence `route[::-1]` yields the complete, correctly spliced itinerary starting from `"JFK"`.

### Hierholzer's Algorithm Implementation

"""

if pos_start_en != -1 and pos_code_en != -1:
    en = en[:pos_start_en] + euler_replacement_en + en[pos_code_en:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Successfully updated English Module 8!")
else:
    print(f"Could not find positions: pos_start={pos_start_en}, pos_code={pos_code_en}")

