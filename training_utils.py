import torch


def get_grad_norm(model: torch.nn.Module) -> float:
    all_grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
    if not all_grads:
        return 0.0
    return torch.norm(torch.cat(all_grads)).item()


def get_param_norm(model: torch.nn.Module) -> float:
    all_params = [p.view(-1) for p in model.parameters()]
    return torch.norm(torch.cat(all_params)).item()


def get_avg_lr(optimizer: torch.optim.Optimizer) -> float:
    return sum(g['lr'] for g in optimizer.param_groups) / len(optimizer.param_groups)


def step_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    metric: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()
