import numpy as np


class Som:
    def __init__(self, size, inner_dim):
        if type(inner_dim) == int:
            inner_dim = (inner_dim,)
        self.inner_dim = inner_dim
        if type(size) == int:
            size = (size, size)
        som_dim = size + inner_dim
        self.som = np.zeros(som_dim)

    def classify(self, example):
        assert self.inner_dim == example.shape
        best_node = None
        best_distance = None
        for i in range(self.som.shape[0]):
            for j in range(self.som.shape[1]):
                dist = np.linalg.norm(self.som[i][j] - example)
                if best_distance is None or dist < best_distance:
                    best_distance = dist
                    best_node = (i, j)
        return best_node


if __name__ == '__main__':
    s = Som((3, 2), 7)
    print(s.som)