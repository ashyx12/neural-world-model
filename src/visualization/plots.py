import matplotlib.pyplot as plt

def plot_rollout_error(errors, horizons=(1, 5, 10, 25)):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(horizons), errors, marker="o")
    ax.set_xlabel("Rollout horizon")
    ax.set_ylabel("Prediction MSE")
    ax.set_title("Compounding prediction error")
    ax.grid(True)
    return fig
