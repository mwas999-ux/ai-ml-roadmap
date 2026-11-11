import numpy as np

# A tiny 6x6 "image": dark on the left (0), bright on the right (9)
image = np.array([
    [0, 0, 0, 9, 9, 9],
    [0, 0, 0, 9, 9, 9],
    [0, 0, 0, 9, 9, 9],
    [0, 0, 0, 9, 9, 9],
    [0, 0, 0, 9, 9, 9],
    [0, 0, 0, 9, 9, 9],
])

# A 3x3 filter that looks for a dark-to-bright vertical edge
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

# Slide the 3x3 "flashlight" across the image
output = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        patch = image[i:i+3, j:j+3]
        output[i, j] = (patch * kernel).sum()

print(output)