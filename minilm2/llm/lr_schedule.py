from typing import Callable
from math import cos, pi

def get_lr_schedule(max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return max_lr * 0.2 + max_lr * 0.8 * step / warmup_steps
        elif step < total_steps:
            # (warmup_steps, total_steps) -> (0, 1) -> (0, pi) -cos-> (0, 1) -> (min_lr, max_lr)
            return (cos((step - warmup_steps) / (total_steps - warmup_steps) * pi) + 1) / 2 * (max_lr - min_lr) + min_lr
        else:
            return min_lr

    return lr_schedule

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    lr_schedule = get_lr_schedule(max_lr=0.1, min_lr=0.001, warmup_steps=10, total_steps=10000)
    x = [i for i in range(12000)]
    y = [lr_schedule(i) for i in x]
    plt.plot(x, y)
    plt.show()
