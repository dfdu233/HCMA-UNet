from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")

        # When resuming from checkpoints, scheduler state can be ahead of max_steps.
        # Clamp to keep the polynomial base in [0, 1] and avoid complex LRs.
        clamped_step = min(max(int(current_step), 0), int(self.max_steps))
        base = 1 - clamped_step / self.max_steps
        new_lr = float(self.initial_lr * (base ** self.exponent))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
