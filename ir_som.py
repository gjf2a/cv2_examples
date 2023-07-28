from typing import List

from som import Som
import numpy as np


def ir_readings_from(filename: str) -> List:
    return [np.asarray(list(eval(line))) for line in open(filename)]


def train_som_from(filename: str, som_side: int, epochs: int) -> Som:
    readings = ir_readings_from(filename)
    ir_som = Som(som_side, 7)
    for epoch in range(epochs):
        learning_rate = (epochs - epoch) / epochs
        radius = som_side * learning_rate
        for reading in readings:
            ir_som.train(reading, learning_rate, radius)
    return ir_som

