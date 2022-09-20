from functools import reduce

import torch

model = torch.load("model_eval.pt")

trainable_params = 0
other_params = 0

for param in model.parameters():
    params_in_current = reduce(lambda a, b: a * b, param.size())
    if param.requires_grad:
        trainable_params += params_in_current
    else:
        other_params += params_in_current

print("================== Parameters ==================")
print("Total parameter count: ", trainable_params + other_params)
print("Trainable parameter count: ", trainable_params)
print("Other (non-trainable) parameter count: ", other_params)


