"""
    Diffusion models as in Denoising Diffusion Probabilistic Models, Ho et al. (2020)
    using the cosine scheduling of Improved Denoising Diffusion Probabilistic Models, Nichol et al. (2021)

    The algorithm is applied to a simple 2D dataset.
    In the end a video will be created which shows the reverse diffusion process from t=T to t=1 for N datapoints
"""

import torch
from matplotlib import animation
from torch.nn import Linear, Tanh, GroupNorm
from torch.nn.modules import Sequential
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# === Hyper parameters for this experiment ===
# Hidden layer size
from schedules.cosine_schedule import CosineSchedule
from schedules.linear_schedule import LinearSchedule
from sinusoidal_embedding import SinusoidalEmbedding

h_size = 64
# The (x,y) coordinates of the "modes" in the dataset. 2D normal distributions will be generated around these modes
modes = [(1, -1), (-1, -1), (0, 1)]
# The standard deviation of the normal distributions
mode_std = 0.1
# The number of samples per mode
n_samples_per_mode = 1000
# Epochs to train for (an epoch is 1 training run over the dataset)
epochs = 300
# Minibatch size
batch_size = 64
# Learning rate
lr = 0.0003
# Number of timesteps T
T = 1000
# Number of evaluation samples
n_eval_samples = 1000
# Use cosine schedule (instead of linear, should improve training)
use_cosine_schedule = True
# Encode the timestep in a sinusoidal embedding (instead of just dumping the normalized value straight into the NN)
use_sin_embedding = True

# === Noise schedule ===
if use_cosine_schedule:
    sched = CosineSchedule(T)
else:
    sched = LinearSchedule(T)


# === Define the prediction model as a simple neural network ===
class PredModel2D(torch.nn.Module):
    def __init__(self, h_size, use_embedding=False, embedding_size=32):
        super().__init__()
        t_size = embedding_size if use_embedding else 1
        self.model = Sequential(
            # 3 inputs, 2 for data and 1 for t
            Linear(2 + t_size, h_size),
            Tanh(),

            Linear(h_size, h_size),
            Tanh(),

            Linear(h_size, 2),
        )

        self.sin_embedding = None
        if use_embedding:
            self.sin_embedding = SinusoidalEmbedding(embedding_size, T, 1.0)

    def forward(self, inp, t):
        if self.sin_embedding is not None:
            t = self.sin_embedding(t)

        return self.model(torch.cat([inp, t], dim=1))


model = PredModel2D(h_size, use_embedding=use_sin_embedding)

opt = RMSprop(model.parameters(), lr)


# === Model sigma at time t ===
def sigma(t):
    sigm = sched.get_betas(t)
    sigm[t <= 1] = 0.0
    return torch.sqrt(sigm)


# === Generate the data ===
data = []
for (mode_x, mode_y) in modes:
    mode_data = torch.stack(
        [
            torch.normal(mode_x, mode_std, size=(n_samples_per_mode,)),
            torch.normal(mode_y, mode_std, size=(n_samples_per_mode,)),
        ], dim=1
    )
    data.append(mode_data)
data = torch.cat(data, dim=0)

dataset = TensorDataset(data)
# === Training ===

dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

for epoch in range(epochs):
    for x, in dataloader:
        opt.zero_grad()
        t = torch.floor(torch.rand((batch_size, 1)) * (T - 1) + 1)
        eps = torch.normal(0, 1, (batch_size, 2))
        alpha_cumulative_t = sched.get_alpha_hats(t)

        x_after = torch.sqrt(alpha_cumulative_t) * x
        x_after += torch.sqrt(1.0 - alpha_cumulative_t) * eps
        prediction = model(x_after, t / T)

        loss = torch.sum(torch.square(eps - prediction)) / batch_size

        loss.backward()
        opt.step()
    loss_v = loss.detach().item()
    print(f"Epoch {epoch}/{epochs}, loss={loss_v}")

# === Evaluation ===
x_T = torch.normal(0, 1, (n_eval_samples, 2))
ones = torch.ones((n_eval_samples, 1), dtype=torch.float32)
xs = []
x_t = x_T
for t_val in range(T, 0, -1):
    t = ones * t_val
    alpha_cumulative_t = sched.get_alpha_hats(t)
    alpha_cumulative_t[alpha_cumulative_t == 0.0] = 1e-6
    beta = sched.get_betas(t)

    pred = model(x_t, t / float(T))
    x_prev = (x_t - (beta / torch.sqrt(1.0 - alpha_cumulative_t)) * pred) / (torch.sqrt(1.0 - beta))
    x_prev += sigma(t) * torch.randn_like(x_t)

    # Set x t to x t-1 (and clamp the values to known ranges so everything stays in line
    x_t = torch.clamp(x_prev, -5, 5)
    xs.append(x_t.detach().numpy())

x0 = x_t

# === Print predictions and create animation ===
print(x0)

fig, ax = plt.subplots()
scatter = ax.scatter(x_T[:, 0].detach().numpy(), x_T[:, 1].detach().numpy())


def update(i):
    scatter.set_offsets(xs[i * 10])


ani = animation.FuncAnimation(fig, update, interval=33, frames=100)
ani.save("diffusion.gif", writer="imagemagick")
plt.show()
