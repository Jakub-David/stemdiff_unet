import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import io


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