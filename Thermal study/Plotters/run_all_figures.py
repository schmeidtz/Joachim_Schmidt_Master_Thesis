"""
run_all_figures.py
==================
Run all standalone journal figure scripts in sequence.

Usage (from any directory):
    python3 Figures_Journal/scripts/run_all_figures.py

Or from inside the scripts folder:
    python3 run_all_figures.py

Prerequisites
-------------
1. Run prepare_exports.py from inside the notebook first:
       %run Figures_Journal/scripts/prepare_exports.py
   This exports the following files to Figures_Journal/data_exports/:
       processed_results.pkl
       surf_fields.pkl
       spec_fields.pkl
       profile_per_spec.pkl
       metrics_df.csv
       prof_df.csv

2. Place TABLE_thermal_expansion.csv in this scripts folder.

Output
------
All figures are saved as .pdf, .svg, and .png to:
    Figures_Journal/
"""

import sys
import time
import traceback
import runpy
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# ── Figure scripts in logical order ───────────────────────────────────────────
FIGURE_SCRIPTS = [
    "fig01_surface_Tmax_evolution.py",
    "fig02_strain_vs_temperature.py",
    "fig03_ext_strain_zz.py",
    "fig03_strain_vs_local_T.py",
    "fig04_ext_strain_yy.py",
    "fig05_cte_vs_temperature.py",
    "fig06_cte_glassy_bars.py",
    "fig07_cte_rubbery_bars.py",
    "fig08_tyt_heatmaps.py",
    "fig09_normalized_temp_profiles.py",
    "fig10_thermal_conductivity_k.py",
    "fig11_individual_strain_cte.py",
    "fig12_delta_T_bars.py",
    "fig_anova_hc3.py",
    "fig_specimen_coord_cte.py",
    "fig_table_specimen_coord.py",
    # ── Specimen-coordinate versions ──────────────────────────────────────────
    "figSC02_strain_vs_temperature.py",
    "figSC03_ext_strain_zz.py",
    "figSC04_ext_strain_yy.py",
    "figSC_ext_strain_directions.py",
    "figSC05_cte_vs_temperature.py",
    "figSC06_cte_glassy_bars.py",
    "figSC07_cte_rubbery_bars.py",
    "figSC08_tyt_heatmaps.py",
    "figSC09_normalized_temp_profiles.py",
    "figSC10_thermal_conductivity_k.py",
    "figSC11_individual_cte.py",
    "figSC12_delta_T_bars.py",
    "figSC_anova_hc3.py",
]

# ── Runner ────────────────────────────────────────────────────────────────────
def main():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    total   = len(FIGURE_SCRIPTS)
    passed  = []
    failed  = []

    print(f"\n{'='*60}")
    print(f"  Journal figure runner  —  {total} scripts")
    print(f"{'='*60}\n")

    for i, script_name in enumerate(FIGURE_SCRIPTS, start=1):
        script_path = SCRIPTS_DIR / script_name
        prefix = f"[{i:2d}/{total}]  {script_name}"

        if not script_path.exists():
            print(f"{prefix}  →  SKIPPED (file not found)")
            failed.append((script_name, "file not found"))
            continue

        print(f"{prefix}  …", end="", flush=True)
        t0 = time.perf_counter()
        try:
            runpy.run_path(str(script_path), run_name="__main__")
            elapsed = time.perf_counter() - t0
            print(f"  OK  ({elapsed:.1f} s)")
            passed.append(script_name)
        except FileNotFoundError as e:
            elapsed = time.perf_counter() - t0
            print(f"  SKIPPED  ({elapsed:.1f} s)\n    ↳ {e}")
            failed.append((script_name, str(e)))
        except Exception:
            elapsed = time.perf_counter() - t0
            print(f"  FAILED  ({elapsed:.1f} s)")
            tb = traceback.format_exc()
            for line in tb.strip().splitlines():
                print(f"    {line}")
            failed.append((script_name, tb.splitlines()[-1]))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Results: {len(passed)}/{total} succeeded")
    if failed:
        print(f"\n  Failed / skipped:")
        for name, reason in failed:
            print(f"    ✗  {name}")
            print(f"       {reason}")
    else:
        print("  All figures generated successfully.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
