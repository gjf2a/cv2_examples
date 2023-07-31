# From https://www.perplexity.ai/search/44c411ea-9b6b-4ec9-9eee-5f65e0aad332?s=c

import matplotlib.pyplot as plt
import numpy as np

# Generate random data for histograms
data = [np.random.normal(0, 1, 1000),
        np.random.normal(2, 1, 1000),
        np.random.normal(-2, 1, 1000),
        np.random.normal(0, 2, 1000),
        np.random.normal(2, 2, 1000),
        np.random.normal(-2, 2, 1000)]

# Create a 2x3 grid of subplots for histograms
fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# Iterate over the subplots and plot histograms
for i, ax in enumerate(axs.flatten()):
    ax.hist(data[i], bins=30, density=True)
    ax.set_title(f'Histogram {i+1}')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()