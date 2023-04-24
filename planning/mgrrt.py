import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, q):
        self.q = q
        self.edges = []
        self.parent = None

class MultiGoalRRT:
    def __init__(self, start, goals, obstacles, step_size=0.1, max_iterations=10000, goal_tolerance=0.1):
        self.start = Node(start)
        self.goals = [Node(goal) for goal in goals]
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.goal_tolerance = goal_tolerance
    
    def run(self):
        nodes = [self.start]
        
        for i in range(self.max_iterations):
            # sample a random configuration
            if np.random.uniform() < 0.3:
                q_rand = self.sample_goal()
            else:
                q_rand = self.sample_free()
            
            # find the nearest node in the tree
            nearest_node = min(nodes, key=lambda node: np.linalg.norm(q_rand - node.q))
            
            # extend the tree towards the random configuration
            q_new = self.steer(nearest_node.q, q_rand)
            
            # check for collisions with obstacles
            if not self.check_collision(q_new):
                continue
            
            # add the new node to the tree
            q_near = self.find_nearest(q_new, nodes)
            new_node = Node(q_new)
            new_node.parent = q_near
            q_near.edges.append(new_node)
            nodes.append(new_node)
            
            # check for goal proximity and add to tree
            for goal in self.goals:
                if np.linalg.norm(goal.q - q_new) < self.goal_tolerance:
                    goal.parent = new_node
                    new_node.edges.append(goal)
            
            # check if all goals have been reached
            if all(goal.parent is not None for goal in self.goals):
                break
        
        paths = []
        for goal in self.goals:
            path = []
            node = goal
            
            while node is not None:
                path.insert(0, node.q)
                node = node.parent
            
            paths.append(path)
        
        return paths
    
    def sample_goal(self):
        # sample a random goal configuration
        return [goal.q for goal in self.goals][np.random.choice(range(len(self.goals)))]
    
    def sample_free(self):
        # sample a random free configuration
        q_rand = np.random.rand(2) * 10 - 5
        while not self.check_collision(q_rand):
            q_rand = np.random.rand(2) * 10 - 5
        return q_rand
    
    def steer(self, q_near, q_rand):
        # extend the tree towards the random configuration
        q_new = q_near + self.step_size * (q_rand - q_near) / np.linalg.norm(q_rand - q_near)
        return q_new
    
    def find_nearest(self, q, nodes):
        # find the nearest node in the tree
        nearest_node = min(nodes, key=lambda node: np.linalg.norm(q - node.q))
        return nearest_node
    
    def check_collision(self, q):
        # check for collisions with obstacles
        for obstacle in self.obstacles:
            if np.linalg.norm(q - obstacle) < 1.0:
                return False
        return True

# example usage
start = np.array([-4, -4])
goals = [np.array([4, 4]), np.array([4, -4]), np.array([-4, 4])]
obstacles = [np.array([-3, 3]), np.array([0, 0]), np.array([-3, -3])]
rrt = MultiGoalRRT(start, goals, obstacles)
paths = rrt.run()

# fig, ax = plt.subplots()
# for path in paths:
#     ax.plot([q[0] for q in path], [q[1] for q in path], '-o')
# ax.scatter(start[0], start[1], marker='*', s=200, label='Start')
# for goal in goals:
#     ax.scatter(goal.q[0], goal.q[1], marker='X', s=200, label='Goal')
# for obstacle in obstacles:
#     circle = plt.Circle(obstacle, radius=1.0, color='r')
#     ax.add_artist(circle)
# ax.legend()
# ax.set_xlim(-5, 5)
# ax.set_ylim(-5, 5)
# plt.show()
