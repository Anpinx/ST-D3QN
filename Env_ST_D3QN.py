import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from A_star import AStar

class Environment:
    def __init__(self):
        self.grid_size = 11
        self.state_space = 2 * (1 + 1 + 1 + 4)
        self.action_space = 9
        self.uav_position = [0, 0]
        self.goal_position = [9, 10]
        self.obstacles = self.generate_obstacles()
        self.dynamic_obstacles = self.generate_dynamic_obstacles()
        self.sub_targets = []
        self.sensor_range = 5

        self.fig, self.ax = plt.subplots()
        self.path = []

        self.F = 10
        self.V = 1
        self.eta1 = 0.03
        self.eta2 = 0.6
        self.eta3 = 0.01
        self.R = 5

        self.lambda_d = self.calculate_distance(self.uav_position, self.goal_position)
        self.previous_distance = self.calculate_distance(self.uav_position, self.goal_position)

    def reset(self):
        self.uav_position = [0, 0]
        self.dynamic_obstacles = self.generate_dynamic_obstacles()
        self.calculate_sub_targets()
        self.path = [self.uav_position.copy()]

        self.lambda_d = self.calculate_distance(self.uav_position, self.goal_position)
        self.previous_distance = self.calculate_distance(self.uav_position, self.goal_position)

        state = self.get_state()
        return state

    def step(self, action):
        old_position = self.uav_position.copy()
        self.uav_position = self.move_uav(self.uav_position, action)

        self.update_dynamic_obstacles()

        if self.uav_position != old_position:
            self.path.append(self.uav_position.copy())

        reward = self.calculate_reward(old_position)

        self.previous_distance = self.calculate_distance(self.uav_position, self.goal_position)

        done = self.is_done()

        state = self.get_state()

        info = {}

        return state, reward, done, info

    def move_uav(self, position, action):
        action_dict = {
            0: (-1, -1),
            1: (0, -1),
            2: (1, -1),
            3: (-1, 0),
            4: (1, 0),
            5: (-1, 1),
            6: (0, 1),
            7: (1, 1),
            8: (0, 0)
        }
        move = action_dict[action]
        new_position = [position[0] + move[0], position[1] + move[1]]

        new_position[0] = max(0, min(self.grid_size - 1, new_position[0]))
        new_position[1] = max(0, min(self.grid_size - 1, new_position[1]))

        if new_position in self.obstacles:
            return position
        else:
            return new_position

    def generate_obstacles(self):
        obstacles = [
            [0, 2], [0, 3], [0, 6], [0, 7],
            [1, 9], [1, 0], [2, 0], [2, 5],
            [2, 9], [3, 6], [3, 5], [4, 2],
            [4, 3], [4, 8], [4, 9], [5, 9],
            [5, 6], [5, 0], [6, 0], [6, 3],
            [6, 9], [8, 8], [7, 3], [7, 2],
            [9, 9], [9, 8], [9, 6], [9, 5],
            [9, 4]
        ]
        return obstacles

    def generate_dynamic_obstacles(self):
        dynamic_obstacles = [
            [1, 7],
            [1, 2],
            [6, 6],
            [8, 1]
        ]
        return dynamic_obstacles

    def update_dynamic_obstacles(self):
        for obstacle in self.dynamic_obstacles:
            action = random.choice(range(9))
            new_position = self.move_uav(obstacle, action)
            if new_position not in self.obstacles:
                obstacle[:] = new_position

    def calculate_sub_targets(self):
        astar = AStar(self.grid_size, self.obstacles)
        path = astar.find_path(self.uav_position, self.goal_position)
        if path:
            self.sub_targets = path[1::5]
        else:
            self.sub_targets = []

    def get_state(self):
        state = []
        state.extend(self.uav_position)
        state.extend(self.goal_position)
        if self.sub_targets:
            state.extend(self.sub_targets[0])
        else:
            state.extend([0, 0])

        obstacles_in_range = []
        for obs in self.obstacles + self.dynamic_obstacles:
            if self.calculate_distance(self.uav_position, obs) <= self.sensor_range:
                obstacles_in_range.extend(obs)
        while len(obstacles_in_range) < 8:
            obstacles_in_range.extend([0, 0])
        obstacles_in_range = obstacles_in_range[:8]
        state.extend(obstacles_in_range)

        return np.array(state, dtype=np.float32)

    def calculate_distance(self, pos1, pos2):
        return math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

    def calculate_reward(self, old_position):
        reward = 0.0

        if self.uav_position in self.obstacles or \
           self.uav_position in self.dynamic_obstacles:
            r_step = -self.V * self.F
        else:
            r_step = -self.V

        current_distance = self.calculate_distance(self.uav_position, self.goal_position)
        delta_d = current_distance - self.previous_distance

        if delta_d < 0:
            r_d = -1 - self.eta1 * delta_d
        else:
            r_d = -self.eta2 * math.exp(delta_d / self.lambda_d)

        vector_to_goal = [self.goal_position[0] - old_position[0], self.goal_position[1] - old_position[1]]
        vector_move = [self.uav_position[0] - old_position[0], self.uav_position[1] - old_position[1]]
        angle = self.calculate_angle(vector_move, vector_to_goal)

        if abs(angle) < 90:
            cos_theta = math.cos(math.radians(angle))
            if cos_theta != 0:
                r_a = -self.eta3 / cos_theta
            else:
                r_a = -2
        else:
            r_a = -2

        if self.uav_position == self.goal_position:
            r_goal = self.V * self.F
        else:
            r_goal = 0

        if self.sub_targets:
            current_sub_target = self.sub_targets[0]
            distance_to_sub_target = self.calculate_distance(self.uav_position, current_sub_target)
            if distance_to_sub_target > self.R:
                r_sub_target = 0
            else:
                r_sub_target = -self.V
        else:
            r_sub_target = 0

        reward = r_step + r_d + r_a + r_goal + r_sub_target

        return reward

    def calculate_angle(self, vector1, vector2):
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.hypot(vector1[0], vector1[1])
        magnitude2 = math.hypot(vector2[0], vector2[1])
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        cos_theta = dot_product / (magnitude1 * magnitude2)
        cos_theta = max(min(cos_theta, 1), -1)
        angle = math.degrees(math.acos(cos_theta))
        return angle

    def is_done(self):
        if self.uav_position == self.goal_position:
            return True
        if self.uav_position in self.obstacles or \
           self.uav_position in self.dynamic_obstacles:
            return True
        return False

    def render(self, episode):
        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_xticks(range(0, self.grid_size + 1))
        self.ax.set_yticks(range(0, self.grid_size + 1))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        plt.title(f'Episode {episode + 1}')

        for obs in self.obstacles:
            rect = Rectangle((obs[0], obs[1]), 1, 1, facecolor='black')
            self.ax.add_patch(rect)

        for obs in self.dynamic_obstacles:
            rect = Rectangle((obs[0], obs[1]), 1, 1, facecolor='red')
            self.ax.add_patch(rect)

        rect = Rectangle((self.uav_position[0], self.uav_position[1]), 1, 1, facecolor='green')
        self.ax.add_patch(rect)

        rect = Rectangle((self.goal_position[0], self.goal_position[1]), 1, 1, facecolor='yellow')
        self.ax.add_patch(rect)

        for pos in self.path:
            rect = Rectangle((pos[0], pos[1]), 1, 1, facecolor='green', alpha=0.3)
            self.ax.add_patch(rect)

        plt.pause(0.01)