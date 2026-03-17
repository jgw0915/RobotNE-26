
from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner


class RRTStarImplementation(RRTStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    
    # 1. Initialize graph (visited_nodes)
    # 2. Sample points on 2d space using sample_random_node()
    # 3. Find the nearest point in graph.
    # 4. Extend the branch of the nearest node and check the collision (New node and edge).
        # Let the nearest point in graph be new node's parent
    # 5. **Find the near nodes set of the new node in a distance range.**
    # 6. **Check if the cost of the path is smaller when selecting a different parent for the new node in the near nodes sets**
    # 7. **Check if the cost of path is smaller when selecting the new node as the parent of the node in near nodes set**
    # 8. Add the new node and edge into the graph
    
    def preloop(self):
        self.start_node.parent = None
        self.start_node.cost = 0.0
        self.goal_node.parent = None
        self.goal_node.cost = float("inf")
        self.visited_nodes:set[PathNode] = {self.start_node}

    def step(self):
        # ===== some given data/parameters =====
        self.start_node
        self.goal_node
        self.world_map # bgr
        self.occupancy_map # bool
        self.goal_threshold
        self.step_size # for steer
        self.search_radius # for reparent/rewire
        # ==========
        random_node = self.sample_random_node() # must use this to 
                                                # sample new node

        if not self.visited_nodes:
            return

        # 3) nearest node
        nearest_node = min(
            self.visited_nodes,
            key=lambda node: calculate_node_distance(node, random_node)
        )

        # 4) steer from nearest toward random with step_size
        nearest_coordinates = nearest_node.coordinates
        random_coordinates = random_node.coordinates
        dx = random_coordinates.x - nearest_coordinates.x
        dy = random_coordinates.y - nearest_coordinates.y
        distance_to_random = calculate_node_distance(nearest_node, random_node)
        if distance_to_random == 0:
            return

        move_ratio = min(1.0, self.step_size / distance_to_random)
        new_x = nearest_coordinates.x + dx * move_ratio
        new_y = nearest_coordinates.y + dy * move_ratio
        new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))

        if new_node in self.visited_nodes:
            return
        if not check_inside_map(self.occupancy_map, new_node):
            return
        if not check_collision_free(self.occupancy_map, nearest_node, new_node):
            return

        # 5) near nodes within search_radius (collision-free to new node)
        near_nodes = [
            node for node in self.visited_nodes
            if calculate_node_distance(node, new_node) <= self.search_radius
            and check_collision_free(self.occupancy_map, node, new_node)
        ]

        # 6) choose best parent for new node
        best_parent = nearest_node
        best_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
        for candidate_parent in near_nodes:
            candidate_cost = (
                candidate_parent.cost +
                calculate_node_distance(candidate_parent, new_node)
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_parent = candidate_parent

        new_node.parent = best_parent
        new_node.cost = best_cost

        # 8) add new node before rewiring
        self.visited_nodes.add(new_node)

        # 7) rewire near nodes through new node if cheaper
        for near_node in near_nodes:
            if near_node is best_parent:
                continue
            rewired_cost = new_node.cost + calculate_node_distance(new_node, near_node)
            if rewired_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = rewired_cost

        # terminate when reaching goal region and edge to goal is collision-free
        if (
            calculate_node_distance(new_node, self.goal_node) <= self.goal_threshold and
            check_collision_free(self.occupancy_map, new_node, self.goal_node)
        ):
            self.goal_node.parent = new_node
            self.goal_node.cost = (
                new_node.cost + calculate_node_distance(new_node, self.goal_node)
            )
            self.visited_nodes.add(self.goal_node)
            self.is_done.set()
    
    def postloop(self):
        if self.goal_node.parent is None:
            return [], self.visited_nodes
        return collect_path(self.goal_node), self.visited_nodes