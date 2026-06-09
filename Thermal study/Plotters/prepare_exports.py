"""
prepare_exports.py
==================
Run this script ONCE from within the notebook environment (Jupyter or IPython)
after all processing cells have executed, to export every data object needed
by the standalone figure scripts.

Usage (inside the notebook):
    %run Figures_Journal/scripts/prepare_exports.py

Or add a cell at the end of the notebook:
    exec(open("Figures_Journal/scripts/prepare_exports.py").read())

All outputs land in:
    Figures_Journal/data_exports/
"""

import pickle
from pathlib import Path

OUT = Path("Figures_Journal/data_exports")
OUT.mkdir(parents=True, exist_ok=True)


def _save_pickle(obj, fname):
    with open(OUT / fname, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    print(f"  ✓  {fname}")


def _save_csv(df, fname):
    import pandas as pd
    df.to_csv(OUT / fname, index=False)
    print(f"  ✓  {fname}")


print("Exporting data objects to", OUT.resolve())
print()

# ── processed_results ─────────────────────────────────────────────────────────
if "processed_results" in dir() or "processed_results" in globals():
    _save_pickle(processed_results, "processed_results.pkl")
else:
    print("  ⚠  processed_results not in scope — skipping")

# ── surf_fields (Camera 1 — surface Tmax evolution) ───────────────────────────
if "surf_fields" in dir() or "surf_fields" in globals():
    _save_pickle(surf_fields, "surf_fields.pkl")
else:
    print("  ⚠  surf_fields not in scope — skipping (run RUN_SURFACE_ANALYSIS block)")

# ── spec_fields (Camera 2 — through-thickness T profiles for k) ───────────────
# Built by build_thermal_fields(); may be called thermal_fields in some cells.
# In the clean pipeline it is not stored as a global, so we call the builder.
_sf = None
for _name in ["spec_fields", "thermal_fields"]:
    if _name in dir() or _name in globals():
        _sf = eval(_name)
        break
if _sf is None and "build_thermal_fields" in globals():
    try:
        _pr = globals().get("processed_results")
        _mf = globals().get("manifest_df")
        if _pr is not None and _mf is not None:
            print("  ℹ  calling build_thermal_fields() to generate spec_fields …")
            _sf, _ = build_thermal_fields(_pr, _mf)
        else:
            print("  ⚠  spec_fields not in scope — skipping")
    except Exception as _e:
        print(f"  ⚠  build_thermal_fields() failed: {_e} — skipping")
if _sf is not None:
    _save_pickle(_sf, "spec_fields.pkl")
else:
    print("  ⚠  spec_fields / thermal_fields not in scope — skipping")

# ── profile_per_spec (normalised T profiles for k calculation) ────────────────
if "profile_per_spec" in dir() or "profile_per_spec" in globals():
    _save_pickle(profile_per_spec, "profile_per_spec.pkl")
else:
    print("  ⚠  profile_per_spec not in scope — skipping")

# ── metrics_df ────────────────────────────────────────────────────────────────
if "metrics_df" in dir() or "metrics_df" in globals():
    _save_csv(metrics_df, "metrics_df.csv")
else:
    print("  ⚠  metrics_df not in scope — skipping (run plot_all_results() first)")

# ── prof_df (thermal conductivity) ────────────────────────────────────────────
if "prof_df" in dir() or "prof_df" in globals():
    _save_csv(prof_df, "prof_df.csv")
else:
    print("  ⚠  prof_df not in scope — skipping")

# ── manifest_df (specimen manifest) ───────────────────────────────────────────
if "manifest_df" in dir() or "manifest_df" in globals():
    _save_csv(manifest_df, "manifest_df.csv")
else:
    print("  ⚠  manifest_df not in scope — skipping")

print()
print("Done.  Files in", OUT.resolve())
print("Now run the standalone figure scripts from Figures_Journal/scripts/")
