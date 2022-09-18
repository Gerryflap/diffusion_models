import torch
import torchvision

from image_models import UResNet
from schedules.cosine_schedule import CosineSchedule
from schedules.linear_schedule import LinearSchedule

cuda = True

# model_eval = torch.load("trained_models/ffhq64/model_eval.pt")
model_eval = torch.load("model_eval.pt")
n_eval_samples = 64
# Resolution can be overridden if model provides different info. Only relevant for older models
res = 64
seed = 42
T = 1000
t_start = T
# Pretend the first N steps (T to T-N) are at timestep T-N to avoid over saturated or broken images
# (advised: N=50 at T=1000)
skip_N_t_steps = 1
# Use cosine schedule (instead of linear. Depends on the model you trained (probably))
use_cosine_schedule = True

# === Noise schedule ===
if use_cosine_schedule:
    sched = CosineSchedule(T)
else:
    sched = LinearSchedule(T)


def sigma(t):
    sigm = sched.get_beta_hats(t)
    sigm[t <= 1] = 0.0
    return torch.sqrt(sigm)


if isinstance(model_eval, UResNet):
    print("UResNet detected, extracting resolution info from it...")
    res = 2 ** (model_eval.downscales + 2)

T_as_tensor = torch.ones(1, ) * T
print("Beta: ", sched.get_betas(T_as_tensor))
print("Alpha_hat: ", sched.get_alpha_hats(T_as_tensor))

generator = torch.Generator()
generator.manual_seed(seed)
x_T_eval = torch.normal(0, 1, (n_eval_samples, 3, res, res), generator=generator)
x_T_eval = x_T_eval.cuda()
torch.manual_seed(seed)


# === Eval function definition ===
def evaluate():
    model_eval.eval()
    xlist = [(x_T_eval[0] + 1.0) / 2.0]
    with torch.no_grad():
        ones = torch.ones((n_eval_samples, 1), dtype=torch.float32)
        if cuda:
            ones = ones.cuda()

        x_t = x_T_eval
        for t_val in range(t_start, 0, -1):
            t_val_model = t_val
            if t_val_model > T - skip_N_t_steps:
                t_val_model = T - skip_N_t_steps
            t = ones * t_val_model
            alpha_cumulative_t = sched.get_alpha_hats(t).view(-1, 1, 1, 1)
            beta = sched.get_betas(t).view(-1, 1, 1, 1)
            sigm = sigma(t).view(-1, 1, 1, 1)
            if cuda:
                alpha_cumulative_t = alpha_cumulative_t.cuda()
                beta = beta.cuda()
                sigm = sigm.cuda()

            # model_inp = torch.cat([x_t, t / float(T)], dim=1)
            pred = model_eval(x_t, t / T)
            x_prev = (x_t - (beta / torch.sqrt(1.0 - alpha_cumulative_t)) * pred) / (torch.sqrt(1.0 - beta))
            x_prev = x_prev + sigm * torch.normal(0.0, 1.0, x_t.size(), device=x_t.device)

            print("%d\t%.5f\t%.5f\t%.5f\t%.5f" % (
            t[0, 0].item(), x_prev.mean().item(), x_prev.std().item(), pred.mean().item(), pred.std().item()))
            # Set x t to x t-1 (and some hacks to keep the deviation under control)
            # if (t_val > T-100):
            #     x_prev = (x_prev - x_prev.mean(dim=(1,2,3), keepdim=True)) / x_prev.std(dim=(1,2,3), keepdim=True)
            x_t = x_prev
            if t_val % (T // 10) == 1:
                xlist.append((x_t[0, :] + 1.0) / 2.0)

        x0 = x_t
        x0 = x0.view(-1, 3, res, res)
        x0 = (x0 + 1.0) / 2.0
        # === Print predictions ===
        torchvision.utils.save_image(x0.clamp(0, 1), "eval_result.png")
        torchvision.utils.save_image(torch.stack(xlist, dim=0), "eval_diff.png", nrow=100)


evaluate()
