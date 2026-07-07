from pathlib import Path
import ediff
import matplotlib.pyplot as plt

SHOW_RESULTS = False
THR = 0.005

result_dir = Path("DATA.STEMDIFF/profiles")
result_dir.mkdir(exist_ok=True)

cif_dir = Path("DATA.STEMDIFF/cif")
cifs = {
    "au": "au_9008463.cif",
    "tbf3": "1530594_tbf3.cif",
    "feo": "Fe3O4.cif",
    "feo_shell": "Fe3O4.cif",
    "laf3": "laf3_9008114.cif",
    "gdf3": "1530594_gdf3.cif",
    "tio2-a": "tio2_anatase_9015929.cif",
    "tio2-r": "tio2_rutile_9015662.cif",
}

results = {}

for name, cif in cifs.items():
    XRD_STRUCTURE = cif_dir / cif
    XRD = ediff.pcryst.XRD_polycrystal(
    structure = XRD_STRUCTURE,
    wavelength = 0.71, two_theta_range = (5,100), peak_profile_sigma = 0.03)

    original_diff = XRD.diffractogram

    # Remove small diffractions
    diffs = XRD.diffractions
    maxI = diffs.Ihkl.max()
    diffs.loc[diffs.Ihkl < maxI * THR, "Ihkl"] = 0

    # Recalculate diffractogram
    XRD.diffractogram = XRD._calculate_diffractogram()

    
    new_diff = XRD.diffractogram
    results[name] = new_diff

    if SHOW_RESULTS:
        fig, ax = plt.subplots(1, 2, figsize=(14, 3))
        ax[0].plot(original_diff.q, original_diff.I)
        ax[1].plot(new_diff.q, new_diff.I)
        ax[0].set_xlim(0, 14)
        ax[1].set_xlim(0, 14)
        fig.suptitle(name)
        fig.tight_layout()
        plt.show()

for name, diff in results.items():
    diff.save(result_dir / name)