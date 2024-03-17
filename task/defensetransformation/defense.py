import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch

from sklearn.gaussian_process import GaussianProcessRegressor

POINTS_AMOUNT = 128
SIGMA = 1

data = np.load("task/defensetransformation/data/DefenseTransformationSubmit.npz")

reprezentations = data["representations"]
dim = reprezentations.shape[1]
examples = reprezentations.shape[0]
random_points = np.zeros(examples)
samples = np.zeros((dim, POINTS_AMOUNT))

vector = np.random.randn(dim, 1)
intercept = reprezentations @ vector

models = {}

for i in tqdm(range(dim)):
    rep = reprezentations.T[i]
    x = np.linspace(np.min(rep), np.max(rep), num=POINTS_AMOUNT)
    points = x[None]
    cov = points - points.T
    cov = cov ** 2
    cov = np.exp(-cov / (2 * SIGMA))
    y = np.squeeze(np.random.multivariate_normal(np.zeros(POINTS_AMOUNT), cov, 1))

    # x = torch.tensor(x, dtype=torch.float32)[:, None]
    # y = torch.tensor(y, dtype=torch.float32)[:, None]
    
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    # criterion = torch.nn.MSELoss()

    # mse = []

    # for epoch in range(100):
    #     optimizer.zero_grad()
    #     outputs = net(x)
    #     loss = criterion(outputs, y)
    #     loss.backward()
    #     optimizer.step()

    #     mse.append(loss.detach().item())

    # if i == 1:
    #     outputs = net(x).squeeze().detach().numpy()
    #     x = x.squeeze().detach().numpy()
    #     y = y.squeeze().detach().numpy()
        
    #     plt.plot(mse)
    #     plt.show()


    models[i] = GaussianProcessRegressor().fit(x[:, None], y[:, None])
    if i == 0:
        plt.plot(x, y, label="real")
        plt.plot(x, models[i].predict(x[:, None]).squeeze(), label="predicted")
        plt.legend()
        plt.show()

print(reprezentations.shape)
print(samples.shape)

# reprezentations = torch.tensor(reprezentations, dtype=torch.float32)

with torch.no_grad():
    for i in models:
        reprezentations[:, i] += models[i].predict(reprezentations[:, i, None]).squeeze()

reprezentations += intercept

print(samples.shape)
print(reprezentations.shape)


np.savez("task/defensetransformation/model/defense_8.npz", representations=reprezentations)
print(reprezentations.shape)
# print(data["labels"].shape)
