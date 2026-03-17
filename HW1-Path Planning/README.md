# HW1 Path Planning: Implementation Notes

## A* Algorithm Implementation

### Pseudocode

**preloop (one-time init)**

- Initialize data structures:
	- `open` as a min-heap of `(f, node)`; start with `(h(start), start)`.
	- `g = {start: 0}`, `h = {start: heuristic(start, goal)}`.
	- `parent = {start: None}`.
	- `closed = set()`.
	- `visited = {start}` (for visualization).
- Don’t expand neighbors here; just seed the frontier.

**step (single A* expansion)**

- If `open` is empty: set `is_done` and store failure state.
- Pop `(f, current)` with the lowest `f` from `open`.
- If `current` already in `closed`, skip to next pop (loop step again).
- Add `current` to `closed`.
- Goal check: if `distance(current, goal) <= goal_threshold`, set `is_done` and mark `goal_node = current`; return.
- For each `nbr` in `get_neighbor_nodes(current)`:
	- Skip if outside map, collides (`occupancy_map` or `check_collision_free`).
	- `tentative_g = g[current] + cost(current, nbr)` (cost can be grid_size for 4-neighbors, or Euclidean if diagonals are allowed).
	- If `nbr` not in `g` or `tentative_g < g[nbr]`:
		- `parent[nbr] = current`
		- `g[nbr] = tentative_g`
		- `h[nbr] = heuristic(nbr, goal)` (Euclidean)
		- Push `(g[nbr] + h[nbr], nbr)` to `open`
		- Add `nbr` to `visited` for visualization

**postloop (finalize result)**

- If goal reached: `path = collect_path(goal_node)` using `parent` links (or helper) and return `(path, visited)`.
- If failure: return `(empty path or None, visited)`.

### What I Learned

- The key to A* performance is maintaining consistent bookkeeping (`g`, `h`, `parent`, `closed`) and only updating a node when a strictly better `g` is found.
- Separating the algorithm into preloop/step/postloop makes debugging much easier and supports step-by-step visualization.
- Collision checks and boundary checks are as important as the search logic; many failures come from invalid neighbor expansion rather than heuristic design.
- The heuristic should be admissible and consistent with motion cost; Euclidean distance works well when movement cost is geometric.

## RRT* Algorithm Implementation

### Pseudocode

1. Initialize graph (`visited_nodes`).
2. Sample points in 2D space using `sample_random_node()`.
3. Find the nearest point in graph.
4. Extend from nearest node and check collision (new node and edge).
	 - Let the nearest point in graph be the new node’s parent.
5. Find the near-node set of the new node within a distance range.
6. Check whether path cost is smaller when selecting a different parent for the new node from the near-node set.
7. Check whether path cost is smaller when selecting the new node as parent for nodes in the near-node set (rewiring).
8. Add the new node and edge into the graph.

### What I Learned

- RRT* differs from basic RRT mainly in optimization: choosing a better parent and rewiring nearby nodes can significantly reduce final path cost.
- The neighbor radius is a practical tradeoff: too small gives weak optimization, too large increases computation each iteration.
- Collision-free edge validation is critical because every improvement step (parent change and rewiring) depends on valid local connections.
- Even though RRT* is asymptotically optimal, finite iteration budgets matter in practice; good sampling and parameter tuning strongly affect final quality.
