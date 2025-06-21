from .show_log import load_log
from matplotlib import pyplot as plt

if __name__ == "__main__":
    log_files = {
        'baseline': "models/baseline/train.log",
        'muon': "models/muon/train.log",
        'fp32': "models/fp32/train.log",
    }
    for name, log_file in log_files.items():
        with open(log_file) as f:
            log = load_log(f.readlines())
        plt.plot(log['steps'], log['losses'], label=name)
    plt.legend()
    plt.show()
