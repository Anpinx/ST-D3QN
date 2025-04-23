import heapq

class AStar:
    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size
        self.obstacles = obstacles

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, goal):
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            neighbors = self.get_neighbors(current)
            for neighbor in neighbors:
                tentative_g_score = g_score[tuple(current)] + 1
                if tuple(neighbor) not in g_score or tentative_g_score < g_score[tuple(neighbor)]:
                    came_from[tuple(neighbor)] = current
                    g_score[tuple(neighbor)] = tentative_g_score
                    f_score[tuple(neighbor)] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[tuple(neighbor)], neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while tuple(current) in came_from:
            current = came_from[tuple(current)]
            path.append(current)
        path.reverse()
        return path

    def get_neighbors(self, node):
        directions = [(-1, -1), (0, -1), (1, -1),
                      (-1, 0),        (1, 0),
                      (-1, 1),  (0, 1),  (1, 1)]
        neighbors = []
        for d in directions:
            neighbor = [node[0] + d[0], node[1] + d[1]]
            if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                if neighbor not in self.obstacles:
                    neighbors.append(neighbor)
        return neighbors