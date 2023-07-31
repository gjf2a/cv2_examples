from typing import List

from som import Som
import numpy as np
import sys
import matplotlib.pyplot as plt

labels = ['SL', 'L', 'FL', 'FCL', 'FCR', 'FR', 'R']


def ir_readings_from(filename: str) -> List:
    return [np.asarray(list(eval(line))) for line in open(filename)]


def train_som_from(filename: str, som_side: int, epochs: int) -> Som:
    readings = ir_readings_from(filename)
    result = Som(som_side, 7)
    for epoch in range(epochs):
        learning_rate = (epochs - epoch) / epochs
        radius = som_side * learning_rate
        for reading in readings:
            result.train(reading, learning_rate, radius)
    return result


def show_ir_som(trained_ir_som: Som):
    width, height = trained_ir_som.size
    fig, axs = plt.subplots(width, height, figsize=(10,6))
    for i, ax in enumerate(axs.flatten()):
        x = i % width
        y = i // width
        ax.bar(labels, trained_ir_som.som[x][y])
        ax.set_title(f'({x}, {y})')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 ir_som.py filename som_side epochs")
    else:
        trained = train_som_from(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
        show_ir_som(trained)
