import numpy as np
import matplotlib.pyplot as plt

def show_diffractograms(original, enhanced, sample=None):
    """
    Display diffractograms (diffraction patterns) of original and enhanced 
    images side by side.

    Parameters
    ----------
    self : object
        Instance of the class containing the show_diffractograms method.
    original : numpy.ndarray
        Original diffractogram data.
    enhanced : numpy.ndarray
        Enhanced diffractogram data.
    sample : str or None, optional
        Description of the sample being analyzed. If provided, it will 
        be included in the titles of the displayed images. Default is None.

    Returns
    -------
    None
        This method displays diffractograms of the original and enhanced 
        images side by side using matplotlib subplots.
    """
    
    diff1 = np.copy(original)
    diff2 = np.copy(enhanced)
    
    thr = 0.08*np.max(diff1)
    diff1[diff1>thr]=thr

    thr = 0.05*np.max(diff2)
    diff2[diff2>thr]=thr
    
    fig, axs = plt.subplots(1,2, figsize=(16,9))
    
    axs[0].imshow(diff1)
    if sample is not None:
        tit1 = "Original image "+sample
        tit2 = "Enhanced image "+sample
    else: 
        tit1 = "Original image"
        tit2 = "Enhanced image"

    axs[0].set_title(tit1,fontsize=16)
    axs[0].axis("off")

    axs[1].imshow(diff2)
    axs[1].set_title(tit2,fontsize=16)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def show_1D_profiles(original, enhanced, idx=None, sample=None):
    """
    Display 1D sum-profiles of original and enhanced images.

    Parameters
    ----------
    original : numpy.ndarray
        Original image data.
    enhanced : numpy.ndarray
        Enhanced image data.
    sample : str or None, optional
        Description of the sample being analyzed. If provided, it will
        be included in the title of the plot. Default is None.

    Returns
    -------
    None
        This function displays the 1D sum-profiles of the original and
        enhanced images using matplotlib.

    """
    # Compute 1D sum-profiles
    o1d = np.sum(original, axis=0)
    e1d = np.sum(enhanced, axis=0)

    # Set up the plot
    plt.figure(figsize=(10,5))
    
    # Plot the 1D sum-profiles
    plt.plot(o1d, label="Original", color='blue', linewidth=3)
    plt.plot(e1d, label="Enhanced", color='orange', linewidth=3)
    

    # Set title and labels
    title = "1D Sum-Profile"
    if sample:
        title += f" of {sample}"
    if idx:
        title += f" , idx={idx}"
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Intensity")

    # Add legend
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()