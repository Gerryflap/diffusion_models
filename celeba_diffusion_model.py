"""
    Diffusion models as in Denoising Diffusion Probabilistic Models, Ho et al. (2020)
    using the cosine scheduling of Improved Denoising Diffusion Probabilistic Models, Nichol et al. (2021)

    The algorithm is applied to the CelebA dataset.
"""

import torch
import torchvision.utils
from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, Compose, CenterCrop, Resize, Lambda

import diffusion_models as dm

# === Hyper parameters for this experiment ===
# Hidden layer size
from celeba_models import UNetCelebA, UNetCelebA32

# CelebA path
from timer import Timer
from util import parameter_ema

path = "data"
h_size = 128
# Epochs to train for (an epoch is 1 training run over the dataset)
epochs = 200
# Minibatch size
batch_size = 64
# Learning rate
lr = 0.0001
# Number of timesteps T
T = 1000
# Number of evaluation samples
n_eval_samples = 64
# Use cuda
cuda = True
# use 32x32 instead of 64x64
use32 = False
# Exponential moving averaging rate over model weights (default 0.9999 in paper)
ema_rate = 0.9999

# === Define the prediction model ===
if use32:
    model = UNetCelebA32(h_size)
    model_eval = UNetCelebA32(h_size)
else:
    model = UNetCelebA(h_size)
    model_eval = UNetCelebA(h_size)

if cuda:
    model = model.cuda()
    model_eval = model_eval.cuda()

parameter_ema(model_eval, model, True)

opt = RMSprop(model.parameters(), lr)
res = 32 if use32 else 64


# === Model sigma at time t ===
def sigma(t):
    sigm = dm.beta(t, T)
    sigm[t <= 1] = 0.0
    return torch.sqrt(sigm)


# === Training ===
dataset = CelebA(path,
                 transform=Compose([
                     CenterCrop(178),
                     Resize(res),
                     ToTensor(),
                     Lambda(lambda tensor: tensor * 2.0 - 1.0)
                 ]),
                 download=True)
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True, num_workers=4)

x = dataloader.__iter__().__next__()[0]
torchvision.utils.save_image((x + 1.0) / 2.0, "results/original.png")
print(x.size())


# === Eval function definition ===
def evaluate(epoch=None):
    model_eval.eval()
    with torch.no_grad():
        x_T = torch.normal(0, 1, (n_eval_samples, 3, res, res))
        ones = torch.ones((n_eval_samples, 1), dtype=torch.float32)
        if cuda:
            x_T = x_T.cuda()
            ones = ones.cuda()

        x_t = x_T
        for t_val in range(T, 1, -1):
            t = ones * t_val
            alpha_cumulative_t = dm.alpha_hat(t, T).view(-1, 1, 1, 1)
            alpha_cumulative_t[alpha_cumulative_t == 0.0] = 1e-6
            beta = dm.beta(t, T).view(-1, 1, 1, 1)
            sigm = sigma(t).view(-1, 1, 1, 1)
            if cuda:
                alpha_cumulative_t = alpha_cumulative_t.cuda()
                beta = beta.cuda()
                sigm = sigm.cuda()

            # model_inp = torch.cat([x_t, t / float(T)], dim=1)
            pred = model_eval(x_t, t / float(T))
            x_prev = (x_t - (beta / (torch.sqrt(1.0 - alpha_cumulative_t))) * pred) / (torch.sqrt(1.0 - beta))
            x_prev += sigm * torch.randn_like(x_t)

            # Set x t to x t-1 (and clamp the values to known ranges so everything stays in line
            x_t = torch.clamp(x_prev, -1000, 1000)

        x0 = x_t
        x0 = x0.view(-1, 3, res, res)
        x0 = (x0 + 1.0) / 2.0
        # === Print predictions and create animation ===
        if epoch is None:
            torchvision.utils.save_image(x0.clamp(0, 1), "results/celeba_trained_final.png")
            torchvision.utils.save_image(x0, "results/norm_celeba_trained_final.png", normalize=True)

        else:
            torchvision.utils.save_image(x0.clamp(0, 1), "results/celeba_trained_epoch_%05d.png" % epoch)
            torchvision.utils.save_image(x0, "results/norm_celeba_trained_epoch_%05d.png" % epoch, normalize=True)
        model_eval.train()


timer = Timer(False)
timer.set()
try:
    for epoch in range(epochs):
        model.train()
        for x, _ in dataloader:
            timer.log_and_set("loading")
            opt.zero_grad()
            t = torch.floor(torch.rand((batch_size, 1)) * (T - 1) + 1)
            eps = torch.normal(0, 1, (batch_size, 3, res, res))
            alpha_cumulative_t = dm.alpha_hat(t, T).view(-1, 1, 1, 1)
            timer.log_and_set("initializing")
            if cuda:
                x = x.cuda()
                t = t.cuda()
                eps = eps.cuda()
                alpha_cumulative_t = alpha_cumulative_t.cuda()
            timer.log_and_set("converting")

            x_after = torch.sqrt(alpha_cumulative_t) * x
            x_after += torch.sqrt(1.0 - alpha_cumulative_t) * eps
            # model_inp = torch.cat([x_after, t / T], dim=1)
            prediction = model(x_after, t / T)

            timer.log_and_set("predicting")

            loss = torch.mean(torch.square(eps - prediction))

            loss.backward()
            opt.step()

            parameter_ema(model_eval, model, rate=ema_rate)

            timer.log_and_set("optimizing")
        loss_v = loss.detach().item()
        print(f"Epoch {epoch}/{epochs}, loss={loss_v}")
        if epoch % 1 == 0:
            evaluate(epoch)
            print("Updated output image")

except KeyboardInterrupt:
    print("Training interrupted, showing results")

# === Evaluation ===
evaluate()
