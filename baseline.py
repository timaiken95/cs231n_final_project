import numpy as np

scores = []
mean = 5.150926126836967

with open("photonet_dataset.txt") as f:
    for l in f.readlines():
    	scores.append((float(l.split()[3]) - mean)**2)


scores = np.array(scores)
print(len(scores))
print(np.sqrt(np.mean(scores)))