import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import io
from loss import prepare_profiles


def create_profile_img(x, y):
    # 1. Create the matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    ax.plot(y.squeeze(), label="Target")
    ax.plot(x.squeeze(), label="NN output")
    ax.set_ylabel("Intensity")
    ax.set_xlabel("Pixels")
    ax.legend()

    # 2. Render the plot to an image buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Convert to a PIL Image or OpenCV array, and then to a NumPy array for logging
    image = PIL.Image.open(buf)
    image_array = np.array(image).transpose(
        2, 0, 1
    )  # Convert to [Channels, Height, Width]
    buf.close()

    plt.close(fig)

    return image_array

def show_diffractograms(imgs, clip_max=None, clip_first=False, title=None):

    fig, axs = plt.subplots(1,len(imgs), figsize=(4 * len(imgs),5))
    if title != None:
        fig.suptitle(title, fontsize=20)
    first = True
    for ax, (name, img) in zip(axs, imgs.items()):
        if clip_max != None and (clip_first or not first):
            img = np.clip(img, 0, clip_max)
        else:
            img = np.clip(img, 0, None)
            img = np.log10(img + 1)
        ax.set_title(name, fontsize=16)
        ax.imshow(img)
        first = False

    plt.tight_layout()

    plt.show()

def show_1D_profiles(imgs, logscale=False):
    # Set up the plot
    plt.figure(figsize=(10,5))

    for name, data in imgs.items():
        if len(data) == 2:
            img, fmt = data
        else:
            img, fmt = data, ""

        # Compute 1D sum-profiles
        if logscale:
            i1d = np.sum(np.log(img + 1), axis=0)
        else:
            i1d = np.sum(img, axis=0)
        
        # Plot the 1D sum-profiles
        plt.plot(i1d, fmt, label=name)

        # plt.plot(o1d, label="Original", color='blue', linewidth=3)
        # plt.plot(t1d, "r--", label="Target", linewidth=2)

    # Set title and labels
    title = "1D Sum-Profile"
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Intensity")

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()