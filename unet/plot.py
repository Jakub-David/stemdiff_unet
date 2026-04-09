import numpy as np
import matplotlib.pyplot as plt

def show_diffractograms(imgs, clip_max=None, clip_first=False):

    fig, axs = plt.subplots(1,len(imgs), figsize=(16,9))
    first = True
    for ax, (name, img) in zip(axs, imgs.items()):
        if clip_max != None and (clip_first or not first):
            img = np.clip(img, 0, clip_max)
        else:
            img = np.clip(img, 0, None)
            img = np.log10(img + 1)
        ax.set_title(name,fontsize=16)
        ax.imshow(img)
        first = False

    plt.axis("off")
    
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