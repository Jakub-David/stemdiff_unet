from pathlib import Path
import stemdiff as sd
import matplotlib.pyplot as plt
import numpy as np

def load_cached(path, name):
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

    dbase = Path(f"dbase/dbase_{name}.zip")

    if not dbase.exists():
        df = sd.dbase.calc_database(SDATA, DIFFIMAGES)
        sd.dbase.save_database(df, dbase)
    else:
        df = sd.dbase.read_database(dbase)

    return SDATA, DIFFIMAGES, df

def plot_stats(SDATA, DIFFIMAGES, df, name):
    df = df.reset_index(drop=True)
    sd.io.set_plot_parameters(size=(18,10), fontsize=11)
    plot = df.plot.line(y=['MaxInt'], color='green')
    plot.set_xlabel('Datafiles')
    plot.set_ylabel('Primary beam intensity')
    plot.set_title(name)
    plot.grid()

    sd.io.set_plot_parameters(size=(18,10), fontsize=11)
    plot = df.plot.line(y=['Xcenter','Ycenter'])
    plot.set_xlabel('Datafiles')
    plot.set_ylabel('XY-position of primary beam')
    plot.set_title(name)
    plot.grid()

    # sd.io.set_plot_parameters(size=(18,10), fontsize=11)
    # plot = df.plot.scatter(x='Peaks', y='S', color='red', marker='x')
    # plot.set_xlabel('Number of peaks')
    # plot.set_ylabel('Shannon entropy')
    # plot.set_title(name)
    # plot.grid()

    sd.io.set_plot_parameters(size=(24,8))
    fig,ax = plt.subplots(nrows=1, ncols=3)
    ax[0] = df.plot.scatter(x='Peaks', y='S', color='red', marker='x', ax=ax[0])
    ax[0].set_xlabel('Number of peaks')
    ax[0].set_ylabel('Shannon entropy')
    ax[1] = df.plot.scatter(x='Peaks', y='MaxInt', color='orange', marker='x', ax=ax[1])
    ax[1].set_xlabel('Number of peaks')
    ax[1].set_ylabel('Maximum intensity')
    ax[1].set_title(name)
    ax[2] = df.plot.scatter(x='S', y='MaxInt', color='orange', marker='x', ax=ax[2])
    ax[2].set_xlabel('Shannon entropy')
    ax[2].set_ylabel('Maximum intensity')
    for i in range(3): ax[i].grid()
    fig.tight_layout()

    # peaks = df.Peaks.unique()
    # peaks.sort()
    # peaks = peaks[[0, len(peaks) // 2, -1]]
    # sd.io.set_plot_parameters(size=(24,7))
    # sd.io.Plots.plot_datafiles_with_NS(
    #     SDATA, df, N=peaks, S=[1] * 3,
    #     icut=200, rsize=120)