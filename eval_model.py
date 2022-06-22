import torch
import torchvision

import diffusion_models as dm

cuda = True

model_eval = torch.load("model_eval.pt")
n_eval_samples = 64
res = 64
x_T_eval = torch.normal(0, 1, (n_eval_samples, 3, res, res))
x_T_eval = x_T_eval.cuda()
T = 1000
t_start = T-1


def sigma(t):
    sigm = dm.beta(t, T)
    sigm[t <= 1] = 0.0
    return torch.sqrt(sigm)


T_as_tensor = torch.ones(1, ) * T
print("Beta: ", dm.beta(T_as_tensor, T))
print("Alpha_hat: ", dm.alpha_hat(T_as_tensor, T))
print("Beta_hat: ", dm.beta_hat(T_as_tensor, T))


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
            t = ones * t_val
            alpha_cumulative_t = dm.alpha_hat(t, T).view(-1, 1, 1, 1)
            beta = dm.beta(t, T).view(-1, 1, 1, 1)
            sigm = sigma(t).view(-1, 1, 1, 1)
            if cuda:
                alpha_cumulative_t = alpha_cumulative_t.cuda()
                beta = beta.cuda()
                sigm = sigm.cuda()

            # model_inp = torch.cat([x_t, t / float(T)], dim=1)
            pred = model_eval(x_t, t / T)
            x_prev = (x_t - (beta / torch.sqrt(1.0 - alpha_cumulative_t)) * pred) / (torch.sqrt(1.0 - beta))
            x_prev = x_prev + sigm * torch.normal(0.0, 1.0, x_t.size(), device=x_t.device)

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
