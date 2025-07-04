from typing import Literal
from matplotlib import pyplot as plt

LabelType = (
    Literal['losses'] |
    Literal['lrs'] |
    Literal['steps'] |
    Literal['timestamps'] |
    Literal['grad_norms'] |
    Literal['val_losses'] |
    Literal['val_steps'] |
    Literal['val_timestamps']
)
LogType = dict[LabelType, list[float] | list[int]]

def load_log(lines: list[str]) -> LogType:
    losses: list[float] = []
    grad_norms: list[float] = []
    timestamps: list[float] = []
    val_losses: list[float] = []
    val_timestamps: list[float] = []
    lrs: list[float] = []
    steps: list[int] = []
    val_steps: list[int] = []
    t0: float | None = None
    for line in lines:
        # loss_type, step, lr, loss, time = line.strip().split(",")
        d = line.strip().split(",")
        if len(d) == 5:
            loss_type, step, lr, loss, time = d
            grad_norm = "0.0"
        elif len(d) == 6:
            loss_type, step, lr, loss, time, grad_norm = d
        elif len(d) == 4:
            loss_type, step, lr, loss = d
            time = step
            grad_norm = "0.0"
        else:
            print(f"Invalid line: {line}")
            continue
        if t0 is None:
            t0 = float(time)
        if loss_type in ("TRAIN", "SFT"):
            losses.append(float(loss))
            lrs.append(float(lr))
            steps.append(int(step))
            timestamps.append(float(time) - t0)
            grad_norms.append(float(grad_norm))
        elif loss_type == "VAL":
            val_losses.append(float(loss))
            val_steps.append(int(step))
            val_timestamps.append(timestamps[-1])
            t0 += float(time) - t0 - timestamps[-1]
    res: LogType = {
        'losses': losses,
        'steps': steps,
        'timestamps': timestamps,
        'val_losses': val_losses,
        'val_steps': val_steps,
        'val_timestamps': val_timestamps,
        'lrs': lrs,
        'grad_norms': grad_norms,
    }
    return res

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m minilm2.utils.show_log <log_file>")
        exit(1)
    log_file = sys.argv[1]
    with open(log_file, 'r') as f:
        lines = f.readlines()
    log = load_log(lines)
    steps = log['steps']
    losses = log['losses']
    val_steps = log['val_steps']
    val_losses = log['val_losses']
    grad_norms = log['grad_norms']
    timestamps = log['timestamps']
    val_timestamps = log['val_timestamps']
    lrs = log['lrs']

    plt.plot(steps, losses, label="train loss")
    plt.plot(val_steps, val_losses, label="val loss")
    plt.plot(steps, grad_norms, label="grad norm")
    plt.xlabel("steps")
    plt.ylabel("loss & grad norm")
    plt.legend()
    plt.show()

    plt.plot(timestamps, losses, label="train loss")
    plt.plot(val_timestamps, val_losses, label="val loss")
    plt.plot(timestamps, grad_norms, label="grad norm")
    plt.xlabel("time/s")
    plt.ylabel("loss & grad norm")
    plt.legend()
    plt.show()
    
    plt.plot(steps, lrs, label="lr")
    plt.xlabel("steps")
    plt.ylabel("learning rate")
    plt.legend()
    plt.show()
