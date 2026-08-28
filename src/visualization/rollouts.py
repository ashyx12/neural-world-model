import matplotlib.pyplot as plt

def plot_frames(frames, title="World-model rollout"):
    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3))
    if len(frames) == 1:
        axes = [axes]
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis("off")
        axes[i].set_title(str(i))
    fig.suptitle(title)
    fig.tight_layout()
    return fig
