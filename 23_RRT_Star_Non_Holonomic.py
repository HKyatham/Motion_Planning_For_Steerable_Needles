import numpy as np
import cv2
import time
import random
import matplotlib.pyplot as plt
import math

# Define a class for the vertex in the RRT tree
class Vertex:
    def __init__(self, x, y, theta) -> None:
        self.x = x  # x-coordinate of the vertex
        self.y = y  # y-coordinate of the vertex
        self.theta = theta  # orientation of the vertex
        self.px = []  # list to store x-coordinates of the path from root to the vertex
        self.py = []  # list to store y-coordinates of the path from root to the vertex
        self.cost = 0  # cost associated with reaching this vertex
        self.parent = None  # parent vertex in the RRT tree

# Define the RRT planner class
class RRTPlanner:
    def __init__(self, start, goal, obstacles, step_size=0.05, clearance=0.05, force=1):
        self.start = Vertex(start[0], start[1], start[2])  # starting vertex
        self.goal = Vertex(goal[0], goal[1], goal[2])  # goal vertex
        self.obstacles = obstacles  # list of obstacles (x, y, size)
        self.step_size = step_size  # step size for extending the tree
        self.clearance = clearance  # clearance from obstacles
        self.force = force  # force parameter
        self.nodes = []  # list to store vertices in the RRT tree
        self.max_iterations = 1000  # maximum number of iterations for planning
        self.path = None  # final path found by the planner
        
    # Function to check collision between a vertex and obstacles
    def check_collision(self, vertex):
        if vertex is None:
            return False

        for (ox, oy, size) in self.obstacles:
            dx_list = [ox - x for x in vertex.px]
            dy_list = [oy - y for y in vertex.py]
            d_list = [dx ** 2 + dy ** 2 for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size + self.clearance) ** 2:
                return False  # collision

        return True
    
    # Function to generate a random vertex in the workspace
    def generate_random_vertex(self):
        angles = [-10, -5, 0, 5, 10]
        if random.randint(0, 100) > 10:
            rnd_vertex = Vertex(
                random.triangular(0.0, 2.0),
                random.triangular(0.0, 2.0),
                random.uniform(-math.pi, math.pi)
            )
        else:  # goal point sampling
            rnd_vertex = Vertex(self.goal.x, self.goal.y, self.goal.theta)

        return rnd_vertex
    
    # Function to find the nearest vertex to a given vertex
    def find_nearest_vertex(self, vertex):
        distances = [(v.x - vertex.x) ** 2 + (v.y - vertex.y) ** 2 for v in self.nodes]
        index = distances.index(min(distances))
        return index
    
    # Function to extend the tree from one vertex towards another
    def extend(self, from_vertex, to_vertex, extend_length=float("inf")):
        new_vertex = Vertex(from_vertex.x, from_vertex.y, from_vertex.theta)
        d, theta = self.calculate_distance_and_angle(new_vertex, to_vertex)

        new_vertex.px = [new_vertex.x]
        new_vertex.py = [new_vertex.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / 0.01)
        if n_expand == 0:
            return None
        da = 20 * math.pi / (180 * n_expand) if (theta - new_vertex.theta) > 0 else -20 * math.pi / (180 * n_expand)
        th = new_vertex.theta

        for _ in range(n_expand):
            new_vertex.x += 0.01 * math.cos(th)
            new_vertex.y += 0.01 * math.sin(th)
            new_vertex.px.append(new_vertex.x)
            new_vertex.py.append(new_vertex.y)
            th += da
            if new_vertex.x == self.goal.x and new_vertex.y == self.goal.y:
                new_vertex.parent = from_vertex
                return new_vertex

        d, _ = self.calculate_distance_and_angle(new_vertex, to_vertex)
        if d <= 0.01:
            new_vertex.px.append(to_vertex.x)
            new_vertex.py.append(to_vertex.y)
            new_vertex.x = to_vertex.x
            new_vertex.y = to_vertex.y

        new_vertex.parent = from_vertex

        return new_vertex
    
    # Function to propagate the cost to leaves in the RRT tree
    def propagate_cost_to_leaves(self, parent_vertex):

        for vertex in self.nodes:
            if vertex.parent == parent_vertex:
                vertex.cost = self.calculate_new_cost(parent_vertex, vertex)
                self.propagate_cost_to_leaves(vertex)
    
    # Function to rewire the tree with a new vertex
    def rewire(self, new_vertex, near_indices):
        for i in near_indices:
            near_vertex = self.nodes[i]
            edge_vertex = self.extend(new_vertex, near_vertex)
            if not edge_vertex:
                continue
            edge_vertex.cost = self.calculate_new_cost(new_vertex, near_vertex)

            no_collision = self.check_collision(edge_vertex)
            improved_cost = near_vertex.cost > edge_vertex.cost

            if no_collision and improved_cost:
                for vertex in self.nodes:
                    if vertex.parent == self.nodes[i]:
                        vertex.parent = edge_vertex
                self.nodes[i] = edge_vertex
                self.propagate_cost_to_leaves(self.nodes[i])
    
    # Function to calculate the distance to the goal from a given point
    def calculate_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.hypot(dx, dy)
    
    # Function to find vertices near a given vertex
    def find_near_vertices(self, new_vertex):
        n_nodes = len(self.nodes) + 1
        r = 500 * math.sqrt(math.log(n_nodes) / n_nodes)
        if hasattr(self, 'expand_dist'):
            r = min(r, 0.3)
        dist_list = [(v.x - new_vertex.x) ** 2 + (v.y - new_vertex.y) ** 2 for v in self.nodes]
        near_indices = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_indices
    
    # Function to calculate distance and angle between two vertices
    def calculate_distance_and_angle(self, from_vertex, to_vertex):
        dx = to_vertex.x - from_vertex.x
        dy = to_vertex.y - from_vertex.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
    
    # Function to calculate new cost for a vertex in the tree
    def calculate_new_cost(self, from_vertex, to_vertex):
        d, a = self.calculate_distance_and_angle(from_vertex, to_vertex)
        return from_vertex.cost + 0.8 * d + 0.2 * (abs(a) * self.force)
    
    # Function to search for the optimal goal vertex
    def search_optimal_goal_vertex(self):

        goal_indices = []
        for (i, vertex) in enumerate(self.nodes):
            if self.calculate_dist_to_goal(vertex.x, vertex.y) <= 0.05:
                goal_indices.append(i)

        final_goal_indices = []
        for i in goal_indices:
            if abs(self.nodes[i].theta - self.goal.theta) <= np.deg2rad(20.0):
                final_goal_indices.append(i)

        if not final_goal_indices:
            return None

        min_cost = min([self.nodes[i].cost for i in final_goal_indices])
        for i in final_goal_indices:
            if self.nodes[i].cost == min_cost:
                return i

        return None
    
    # Function to generate the final path from start to goal vertex
    def generate_final_path(self, goal_index):
        path = [[self.goal.x, self.goal.y]]
        vertex = self.nodes[goal_index]
        while vertex.parent:
            for (x, y) in zip(reversed(vertex.px), reversed(vertex.py)):
                path.append([x, y])
            vertex = vertex.parent
        path.append([self.start.x, self.start.y])
        return path

    # Function to plan the path from start to goal
    def plan(self):
        self.nodes = [self.start]
        for i in range(self.max_iterations):
            print("Iter:", i, ", number of nodes:", len(self.nodes))
            rnd_vertex = self.generate_random_vertex()
            nearest_index = self.find_nearest_vertex(rnd_vertex)
            new_vertex = self.extend(self.nodes[nearest_index], rnd_vertex)
            
            if new_vertex:
                if self.check_collision(new_vertex):
                    near_indices = self.find_near_vertices(new_vertex)
                    self.nodes.append(new_vertex)
                    self.rewire(new_vertex, near_indices)

                if new_vertex:
                    last_index = self.search_optimal_goal_vertex()
                    if last_index:
                        return self.generate_final_path(last_index)

        print("Reached maximum iteration")

        last_index = self.search_optimal_goal_vertex()
        if last_index:
            return self.generate_final_path(last_index)
        else:
            print("Cannot find path")

        return None

    # Function to draw the RRT graph and path
    def draw_graph(self, path=None, rnd=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for vertex in self.nodes:
            if vertex.parent:
                plt.plot(vertex.px, vertex.py, "-g")

        for (ox, oy, size) in self.obstacles:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        if path:
            path_x = [x[0] for x in path]
            path_y = [x[1] for x in path]
            plt.plot(path_x, path_y, color='blue', linewidth=3)
        plt.axis([0, 2.0, 0, 2.0])
        plt.grid(True)
        plt.show()

# Main function
if __name__ == "__main__":
    # Define obstacles in the workspace
    obstacles = [
        (1.8, 1.8, 0.6),
        (1.0, 0.5, 0.2)
    ]
    # Input start and goal coordinates
    start = tuple([float(x) for x in input("Enter the start coordinates like 0.2,0.8,0.2: ").split(',')])
    goal = tuple([float(x) for x in input("Enter the goal coordinates like 1.4,1.0,0.0: ").split(',')])
    clearance = 0.05  # clearance from obstacles
    step_size = 0.05  # step size for extending the tree


    
    # Create an instance of the RRTPlanner class
    planner = RRTPlanner(start, goal, obstacles, step_size=step_size, clearance=clearance)
    # Plan the path
    path = planner.plan()
    # Draw the RRT graph and path
    planner.draw_graph(path)
