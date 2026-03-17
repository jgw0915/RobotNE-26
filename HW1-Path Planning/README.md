# HW1 Path Planning: A* and RRT* Implementation Summary

## A* Algorithm Implementation

### Core Idea

A* performs best-first search by minimizing:

$$
f(n)=g(n)+h(n),
$$

where `g(n)` is the accumulated cost from start to node `n`, and `h(n)` estimates the remaining cost to the goal.

### Pseudocode (preloop / step / postloop)

**Preloop (one-time initialization)**

- Initialize `open` as a min-heap and push `(h(start), start)`.
- Initialize `g`, `h`, and `parent` as `{start:0}`, `{start:heuristic(start,goal)}`, and `{start:None}`.
- Initialize `closed = set()` and `visited = {start}`.

**Step (single expansion)**

- If `open` is empty, terminate with failure; otherwise pop the node with minimum `f`.
- If the node is already in `closed`, skip it; else add it to `closed`.
- If `distance(current, goal) <= goal_threshold`, mark success and stop.
- For each node in `get_neighbor_nodes(current)`:
  - skip invalid or colliding neighbors;
  - compute `tentative_g = g[current] + cost(current,nbr)`;
  - if improved, update `parent/g/h` and push `(g+h,nbr)` into `open`.

**Postloop (finalization)**

- If goal is reached, reconstruct the path by tracing `parent` links.
- Otherwise, return an empty path (or `None`) and `visited`.

### What I Learned from A*

- Correct bookkeeping of `g/h/parent/closed` is essential for correctness.
- The preloop--step--postloop split makes debugging and visualization easier.
- In practice, collision and boundary checks usually determine robustness.

## RRT* Algorithm Implementation

### Core Idea

RRT* grows a feasible tree and improves path cost through parent selection and rewiring.

### Pseudocode (preloop / step / postloop)

**Preloop (one-time initialization)**

- Initialize tree `visited_nodes` with the start node.
- Set planning parameters: maximum iterations, step size, and neighbor radius.
- Define goal-biased sampling in `sample_random_node()`: with probability `0.3`, return the goal node as the sample; otherwise, return a random free-space node.

**Step (single growth and optimization iteration)**

- Sample a node using `sample_random_node()` (goal with probability `0.3`, random free-space node with probability `0.7`), then find its nearest tree node.
- Steer from the nearest node toward the sample; keep the new node only if the edge is collision-free.
- Find nearby nodes within radius and choose the parent with minimum total cost.
- Rewire nearby nodes if connecting through the new node reduces cost and remains collision-free.
- Insert the new node and updated edges into the tree.
- If `distance(new_node, goal_node) <= goal_threshold` and the edge from new node to goal is collision-free, connect the goal node, update its cost/parent, and terminate early with success.

**Postloop (finalization)**

- If the goal node was connected during planning, reconstruct the final trajectory by tracing parent links to the start.
- Otherwise, return failure with the explored tree.

### What I Learned from RRT*

- The main upgrade over RRT is local optimization via parent reselection and rewiring.
- A goal-bias probability of `0.3` accelerates convergence toward the target while still preserving exploration.
- Neighbor radius controls the tradeoff between path quality and runtime.
- With finite iterations, sampling strategy and collision checks dominate final performance.
