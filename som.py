import math

import numpy as np
import unittest


def distance_weight(winning_node, current_node, radius):
    distance = ((current_node[0] - winning_node[0])**2 + (current_node[1] - winning_node[1])**2)**0.5
    return 1.0 - min(1.0, distance / radius)


class Som:
    def __init__(self, size, inner_dim, rng=None):
        if type(inner_dim) == int:
            inner_dim = (inner_dim,)
        self.inner_dim = inner_dim
        if type(size) == int:
            size = (size, size)
        som_dim = size + inner_dim
        if rng is None:
            self.som = np.zeros(som_dim)
        else:
            self.som = np.random.random(som_dim)

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

    def train(self, example, learning_rate, radius):
        best_node = self.classify(example)
        for i in range(self.som.shape[0]):
            for j in range(self.som.shape[1]):
                self.som[i][j] += learning_rate * distance_weight(best_node, (i, j), radius) * (example - self.som[i][j])


class SomTests(unittest.TestCase):
    def test_zeroed(self):
        s = Som(2, 3)
        s.train(np.array([1, 1, 1]), 1.0, 1.0)
        s.train(np.array([-1, -1, -1]), 1.0, 1.0)
        assert s.classify(np.array([1, 1, 1])) == (0, 0)
        assert s.classify(np.array([-1, -1, -1])) == (0, 1)
        print(s.som)


if __name__ == '__main__':
    s = Som((3, 2), 7, np.random.default_rng())
    print(s.som)
