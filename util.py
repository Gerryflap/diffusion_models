import torch.nn


def parameter_ema(eval_model: torch.nn.Module, current_model: torch.nn.Module, initial=False, rate=0.9999):
    with torch.no_grad():
        if initial:
            rate = 0.0

        for ev_param, curr_param in zip(eval_model.parameters(), current_model.parameters()):
            ev_param.data = ev_param.data * rate + curr_param.data * (1.0 - rate)
