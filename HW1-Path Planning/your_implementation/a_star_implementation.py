
import heapq

import cv2
import numpy as np

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    def preloop(self):
        self.open:list[tuple[float, int, PathNode]] = []
        self.open_counter = 0
        self.g:dict[PathNode, float] = {self.start_node: 0.0}
        self.h:dict[PathNode, float] = {
            self.start_node: calculate_node_distance(
                self.start_node,
                self.goal_node
            )
        }
        self.parent:dict[PathNode, PathNode | None] = {self.start_node: None}
        self.closed:set[PathNode] = set()
        self.visited_nodes:set[PathNode] = {self.start_node}
        self.goal_reached = False
        self.goal_node.parent = None
        self.goal_node.cost = 0.0
        self.goal_reached_node:PathNode | None = None

        self.start_node.parent = None
        self.start_node.cost = 0.0
        heapq.heappush(
            self.open,
            (self.h[self.start_node], self.open_counter, self.start_node)
        )
        self.open_counter += 1

    def step(self):
        # to sample neighbor nodes, use self.get_neighbor_nodes(current_node)
        while self.open:
            _, _, current_node = heapq.heappop(self.open)
            if current_node in self.closed:
                continue
            break
        else:
            self.is_done.set()
            return

        self.closed.add(current_node)
        self.visited_nodes.add(current_node)

        if calculate_node_distance(current_node, self.goal_node) <= self.goal_threshold:
            self.goal_reached = True
            self.goal_reached_node = current_node
            self.is_done.set()
            return

        for neighbor_node in self.get_neighbor_nodes(current_node):
            tentative_g = self.g[current_node] + calculate_node_distance(
                current_node,
                neighbor_node
            )

            if neighbor_node in self.g and tentative_g >= self.g[neighbor_node]:
                continue

            self.parent[neighbor_node] = current_node
            self.g[neighbor_node] = tentative_g
            self.h[neighbor_node] = calculate_node_distance(
                neighbor_node,
                self.goal_node
            )
            neighbor_node.parent = current_node
            neighbor_node.cost = tentative_g
            heapq.heappush(
                self.open,
                (
                    tentative_g + self.h[neighbor_node],
                    self.open_counter,
                    neighbor_node
                )
            )
            self.open_counter += 1

    def postloop(self):
        if self.goal_reached and self.goal_reached_node is not None:
            if self.goal_reached_node == self.goal_node:
                self.goal_node.parent = self.goal_reached_node.parent
                self.goal_node.cost = self.goal_reached_node.cost
            else:
                self.goal_node.parent = self.goal_reached_node
                self.goal_node.cost = (
                    self.goal_reached_node.cost +
                    calculate_node_distance(self.goal_reached_node, self.goal_node)
                )
            self.visited_nodes.add(self.goal_node)
            return collect_path(self.goal_node), self.visited_nodes

        return [], self.visited_nodes