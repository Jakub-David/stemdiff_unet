from pathlib import Path
import stemdiff as sd
import ediff as ed
import idiff.bkg2d as bkg
import matplotlib.pyplot as plt
import numpy as np

def eld_to_np(ELD):
    return ELD.diffractogram.loc[:, ("q", "I")].to_numpy().T

def create_profile(arr, XRD, xrange, xrd_range, eld_range=None, show=True, bkg_1d=None,
                   bkgp={}, center=None, in_file="center.txt"):
    ELD = ed.pcryst.ELD_polycrystal(arr)
    # ----- show the experimental 2D-diffractogram
    if show:
        ELD.diffractogram2D.show(icut=50, cmap='viridis')

    ELD.find_center(detection=center, refinement=None, icut=50, verbose=1, in_file=in_file)
    # ----- show the results of center determination
    if show:
        ELD.center.show(csquare=250)

    # [5] ELD :: calculate 1D diffractogram = convert 2D->1D
    # (this sets self.diffractogram => ed.io.Diffractogram1D (with 2 cols)
    ELD.calculate_1Ddiffractogram()
    # ----- show the 1D diffractogram = radially averaged 2D diffractogram
    if show:
        ELD.diffractogram.show('Pixels','Iraw',
        Xlabel='Pixels', Ylabel='Intensity (raw)', Xlim=(0,300), Ylim=200)


    # [6] ELD :: subtract background in 1D diffractogram
    # (this updates self.diffractogram = ed.io.Diffractogram (2 cols => 4 cols)
    ELD.subtract_background(bkg_1d, xrange=xrange, **bkgp)
    # ----- show the results of bkg correction = Ibkg and I = Iraw-Ibkg
    if show:
        ELD.diffractogram.show('Pixels','I',
        Xlabel='Pixels', Ylabel='Intensity (net)')

    # [7] ELD :: calibrate + normalize 1D diffractogram = Pixels->q + normalized I
    # (this updates self.profile => ed.io.Diffractogram object (4 cols => 5 cols)
    ELD.calibrate_and_normalize('MaxPeaksInRange', XRD, xrd_range=xrd_range, eld_range=eld_range)
    # ----- show the calibrated ELD diffractogram + corresponding XRD diffractogram
    if show:
        ELD.diffractogram.show('q','I',
        Xlabel='q [1/A]', Ylabel='Intensity', Xlim=(1.5,10.5))

    return ELD

def preview(SDATA, df, bkgp, nn_path, show_idx, bkgpt=None, rescale=None):
    nn_bkgp = {"path": nn_path}
    for i in show_idx:
        datafile_name = SDATA.data_dir.joinpath(df.iloc[i].DatafileName)
        img = sd.io.Datafiles.read(SDATA, datafile_name)
        if rescale is not None:
            img = sd.io.Arrays.rescale(img, rescale, order=3)
        img_gaussian = bkg.gaussian(img, **bkgp)
        nn = bkg.NeuralNetwork(**nn_bkgp)
        img_nn = nn.predict(img)
        if bkgpt == None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3))
        ax1.imshow(np.log10(img + 1))
        ax1.set_title("Original")
        ax2.imshow(np.log10(img_gaussian + 1))
        ax2.set_title("Gaussian")
        ax3.imshow(np.log10(img_nn + 1))
        ax3.set_title("NN")
        if bkgpt != None:
            img_tophat = bkg.tophat(img, **bkgpt)
            ax4.imshow(np.log10(img_tophat + 1))
            ax4.set_title("tophat")
        fig.tight_layout()
        plt.show()

def filter_datafiles(df, n_datafiles):
    df = df[(df.Peaks > 0)]
    df = df[(df.MaxInt < 11800)]

    x_mean = df.Xcenter.mean()
    y_mean = df.Ycenter.mean()
    df = df[(df.Xcenter - x_mean).abs() < 15]
    df = df[(df.Ycenter - y_mean).abs() < 15]

    df = df.sort_values(by=['Peaks','S'], ascending=[False,False])[0:n_datafiles]

    return df

def load_cached(path, name, db_dir="dbase", db_file=None, calculate_db=True):
    if len(list(path.glob("*.dat"))) > 0:
        pattern = "*.dat"
    else:
        pattern = "??/*.dat"
    SDATA = sd.gvars.SourceData(
        detector  = sd.detectors.TimePix(),
        data_dir  = path,
        filenames = pattern
    )

    DIFFIMAGES = sd.gvars.DiffImages(
        imgsize=256, psfsize=120,
        ctype=2, csquare=20, cintensity=0.8,
        peak_height=100, peak_dist=9
    )

    db_dir = Path(db_dir)
    if db_file is None:
        dbase = Path(db_dir / f"dbase_{name}.zip")
    else:
        dbase = Path(db_dir / db_file)

    if not dbase.exists() and calculate_db:
        print("Warning: dbase not found - calculating new dbase")
        df = sd.dbase.calc_database(SDATA, DIFFIMAGES)
        sd.dbase.save_database(df, dbase)
    else:
        df = sd.dbase.read_database(dbase)

    return SDATA, DIFFIMAGES, df