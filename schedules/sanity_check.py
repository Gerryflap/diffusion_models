import matplotlib.pyplot as plt
import torch

from schedules.cosine_schedule import CosineSchedule
from schedules.linear_schedule import LinearSchedule

ts = torch.linspace(1, 1000, 1000)
lin_sched = LinearSchedule(1000)
cos_sched = CosineSchedule(1000)

# Alpha_hat
plt.plot(ts, lin_sched.get_alpha_hats(ts), label="Linear schedule")
plt.plot(ts, cos_sched.get_alpha_hats(ts), label="Cosine schedule")
plt.legend()
plt.title(r"$\hat{\alpha_t}$ for linear and cosine schedules")
plt.xlabel(r"$t$")
plt.ylabel(r"$\hat{\alpha_t}$")
plt.show()

# Beta
plt.plot(ts, lin_sched.get_betas(ts), label="Linear schedule")
plt.plot(ts, cos_sched.get_betas(ts), label="Cosine schedule")
plt.legend()
plt.title(r"$\beta_t$ for linear and cosine schedules")
plt.xlabel(r"$t$")
plt.ylabel(r"$\beta_t$")
plt.show()
