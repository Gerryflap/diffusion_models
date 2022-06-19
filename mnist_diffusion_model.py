"""
    Diffusion models as in Denoising Diffusion Probabilistic Models, Ho et al. (2020)
    using the cosine scheduling of Improved Denoising Diffusion Probabilistic Models, Nichol et al. (2021)

    The algorithm is applied to the MNIST dataset.
"""

import torch
import torchvision.utils
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import diffusion_models as dm

# === Hyper parameters for this experiment ===
# Hidden layer size
from mnist_models import UNetMNIST

h_size = 8
# The number of samples per mode
n_samples_per_mode = 1000
# Epochs to train for (an epoch is 1 training run over the dataset)
epochs = 200
# Minibatch size
batch_size = 64
# Learning rate
lr = 0.001
# Number of timesteps T
T = 1000
# Number of evaluation samples
n_eval_samples = 64

# === Define the prediction model ===
# model = Sequential(
#     Linear(28*28 + 1, h_size),
#     Tanh(),
#
#     Linear(h_size, 28*28),
# )
model = UNetMNIST(h_size)


opt = RMSprop(model.parameters(), lr)


# === Model sigma at time t ===
def sigma(t):
    sigm = dm.beta(t, T)
    sigm[t <= 1] = 0.0
    return torch.sqrt(sigm)



# === Training ===
dataset = MNIST("data/", train=True, transform=ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

try:
    for epoch in range(epochs):
        for x, _ in dataloader:
            opt.zero_grad()
            t = torch.floor(torch.rand((batch_size, 1)) * (T - 1) + 1)
            eps = torch.normal(0, 1, (batch_size, 1, 28, 28))
            alpha_cumulative_t = dm.alpha_hat(t, T).view(-1, 1, 1, 1)

            x_after = torch.sqrt(alpha_cumulative_t) * x
            x_after += torch.sqrt(1.0 - alpha_cumulative_t) * eps
            # model_inp = torch.cat([x_after, t / T], dim=1)
            prediction = model(x_after, t/T)

            loss = torch.mean(torch.square(eps - prediction))

            loss.backward()
            opt.step()
        loss_v = loss.detach().item()
        print(f"Epoch {epoch}/{epochs}, loss={loss_v}")
except KeyboardInterrupt:
    print("Training interrupted, showing results")

# === Evaluation ===
x_T = torch.normal(0, 1, (n_eval_samples, 1, 28, 28))
ones = torch.ones((n_eval_samples, 1), dtype=torch.float32)
x_t = x_T
for t_val in range(T, 1, -1):
    t = ones * t_val
    alpha_cumulative_t = dm.alpha_hat(t, T).view(-1, 1, 1, 1)
    alpha_cumulative_t[alpha_cumulative_t == 0.0] = 1e-6
    beta = dm.beta(t, T).view(-1, 1, 1, 1)

    # model_inp = torch.cat([x_t, t / float(T)], dim=1)
    pred = model(x_t, t / float(T))
    x_prev = (x_t - (beta / (torch.sqrt(1.0 - alpha_cumulative_t))) * pred) / (torch.sqrt(1.0 - beta))
    x_prev += sigma(t).view(-1, 1, 1, 1) * torch.randn_like(x_t)

    # Set x t to x t-1 (and clamp the values to known ranges so everything stays in line
    x_t = torch.clamp(x_prev, -5, 5)

x0 = x_t
x0 = x0.view(-1, 1, 28, 28)
# === Print predictions and create animation ===
torchvision.utils.save_image(x0, "output_images.png")
