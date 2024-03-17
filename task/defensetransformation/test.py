import numpy as np

load = np.load("task/defensetransformation/model/defense_8.npz")
x = load["representations"]
print(x.dtype)
# np.savez("task/defensetransformation/model/defense_4.npz", representations=load["reprezentations"])
# data = np.load("task/defensetransformation/data/DefenseTransformationSubmit.npz")

# print(load["reprezentations"].shape)
# print(data["representations"].shape)
# print(data["reprezentations"].shape)