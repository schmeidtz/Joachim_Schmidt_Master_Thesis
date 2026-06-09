# # """
# thermal_viewer.py

# Interactive viewer for thermal CSV data from Thermal_Cam_Single_AllPixels.ipynb.

# Controls:
# Mouse hover       — live temperature readout at cursor position
# Click + drag      — draw a line; shows temperature profile along it
# Right-click       — clear the line
# Frame slider      — scrub through time
# Space             — play / pause
# Left / Right      — step one frame
# Scroll wheel      — step one frame
# C                 — clear line
# S                 — save current frame as PNG
# Q / Escape        — quit

# Layout:
# Left              — heatmap with dr/dc axes, hover crosshair, line overlay
# Top-right         — temperature profile along drawn line (or hover column profile)
# Bottom-right      — time-series of hovered pixel (or line mean) across all frames

# Run:  python thermal_viewer.py
# """

# ==========================================

# CONFIGURATION

# ==========================================

CMAP            = 'inferno'
INTERP          = 'lanczos'
TEMP_MIN        = None      # None = auto from data
TEMP_MAX        = None      # None = auto from data
PLAYBACK_FPS    = 10.0      # frames per second during playback
PROFILE_SAMPLES = 200       # number of sample points along a drawn line

# ==========================================

# IMPORTS

# ==========================================

import csv, os, re, sys
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')   # interactive backend — required for mouse/keyboard events
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

FONTSIZE = 24

# ==========================================

# FILE PICKER

# ==========================================

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
CSV_PATH = filedialog.askopenfilename(
    title='Select thermal CSV',
    filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    initialdir='Data Recordings',
)
root.destroy()

if not CSV_PATH:
    print('No file selected – exiting.')
    sys.exit(0)

print(f'Selected: {CSV_PATH}')

# ==========================================

# LOAD CSV

# ==========================================

print('Loading…')

# Read header line only (no full-file load)
with open(CSV_PATH, encoding='utf-8-sig') as f:
    headers = [h.strip() for h in f.readline().rstrip('\r\n').split(';')]

NEW_FORMAT = 'nozzle_cx' in headers
if NEW_FORMAT:
    px_cols = [h for h in headers if re.match(r'px_dr[+-]?\d+_dc[+-]?\d+', h)]
    dr_vals = sorted({int(re.search(r'px_dr([+-]?\d+)_dc', h).group(1)) for h in px_cols})
    dc_vals = sorted({int(re.search(r'px_dr[+-]?\d+_dc([+-]?\d+)', h).group(1)) for h in px_cols})
    ts_col  = 'elapsed_s' if 'elapsed_s' in headers else 'timestamp'
    has_elapsed = 'elapsed_s' in headers
else:
    px_cols = [h for h in headers if re.match(r'px_r\d+_c\d+', h)]
    dr_vals = list(range(len({int(re.search(r'px_r(\d+)_c', h).group(1)) for h in px_cols})))
    dc_vals = list(range(len({int(re.search(r'px_r\d+_c(\d+)', h).group(1)) for h in px_cols})))
    ts_col  = 'timestamp'
    has_elapsed = False

GRID_H = len(dr_vals); GRID_W = len(dc_vals)
DR_MIN, DR_MAX = dr_vals[0], dr_vals[-1]
DC_MIN, DC_MAX = dc_vals[0], dc_vals[-1]
dr_axis = np.array(dr_vals)
dc_axis = np.array(dc_vals)

# Preprocess: convert European CSV to standard CSV for fast C parser
import tempfile
file_size = os.path.getsize(CSV_PATH)
print(f'  Preprocessing {file_size / 1e9:.1f} GB  (comma→dot, semicolon→comma)...')
tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv')
try:
    with open(CSV_PATH, 'rb') as fin, os.fdopen(tmp_fd, 'wb') as fout:
        while True:
            block = fin.read(64 * 1024 * 1024)   # 64 MB blocks
            if not block:
                break
            block = block.replace(b',', b'.').replace(b';', b',')
            fout.write(block)

    # Parse with fast C engine (no decimal kwarg needed)
    print('  Parsing CSV (pandas C engine, chunked)...')
    needed_cols = [ts_col] + px_cols

    chunk_ts, chunk_frames = [], []
    for chunk in pd.read_csv(tmp_path, sep=',',
                              encoding='utf-8-sig', usecols=needed_cols,
                              chunksize=100_000):
        valid = chunk[px_cols[0]].notna()
        if not valid.all():
            chunk = chunk.loc[valid]
        if chunk.empty:
            continue
        chunk_ts.append(chunk[ts_col].values.astype(np.float64))
        chunk_frames.append(
            chunk[px_cols].values.astype(np.float32).reshape(-1, GRID_H, GRID_W))
finally:
    os.unlink(tmp_path)

timestamps = np.concatenate(chunk_ts);     del chunk_ts
frames     = np.concatenate(chunk_frames); del chunk_frames

if not has_elapsed:
    timestamps -= timestamps[0]
N = len(frames)
print(f'  {N} frames, {GRID_H}×{GRID_W} grid, duration {timestamps[-1]:.1f} s')

# Colour scale

all_vals = frames.ravel()
vmin = TEMP_MIN if TEMP_MIN is not None else float(np.nanpercentile(all_vals,  1))
vmax = TEMP_MAX if TEMP_MAX is not None else float(np.nanpercentile(all_vals, 99))
del all_vals


# ==========================================

# HELPER: grid coords from axes coords

# ==========================================

def axes_to_grid(ax_x, ax_y):
    """Convert continuous axes coordinates to nearest grid indices (col, row)."""
    col = int(round((ax_x - DC_MIN) / max(DC_MAX - DC_MIN, 1) * (GRID_W - 1)))
    row = int(round((ax_y - DR_MIN) / max(DR_MAX - DR_MIN, 1) * (GRID_H - 1)))
    return (np.clip(col, 0, GRID_W-1),
            np.clip(row, 0, GRID_H-1))

def grid_to_axes(col, row):
    """Convert grid indices to axes coordinates."""
    ax_x = DC_MIN + col / max(GRID_W-1, 1) * (DC_MAX - DC_MIN)
    ax_y = DR_MIN + row / max(GRID_H-1, 1) * (DR_MAX - DR_MIN)
    return ax_x, ax_y

def sample_line(grid, r0, c0, r1, c1, n=PROFILE_SAMPLES):
    """Sample temperature along a line from (r0,c0) to (r1,c1) in grid coords."""
    rs = np.linspace(r0, r1, n)
    cs = np.linspace(c0, c1, n)
    ri = np.clip(np.round(rs).astype(int), 0, GRID_H-1)
    ci = np.clip(np.round(cs).astype(int), 0, GRID_W-1)
    return grid[ri, ci]

def sample_line_pixels(grid, r0, c0, r1, c1):
    """Return temperatures at each unique pixel the line passes through (no duplicates)."""
    n = max(abs(r1 - r0), abs(c1 - c0)) + 1
    rs = np.clip(np.round(np.linspace(r0, r1, n)).astype(int), 0, GRID_H - 1)
    cs = np.clip(np.round(np.linspace(c0, c1, n)).astype(int), 0, GRID_W - 1)
    mask = np.concatenate([[True], (rs[1:] != rs[:-1]) | (cs[1:] != cs[:-1])])
    return grid[rs[mask], cs[mask]]

def line_distances(r0, c0, r1, c1, n=PROFILE_SAMPLES):
    """Distances along the line in dr/dc units."""
    dr = (r1 - r0) / max(GRID_H-1, 1) * (DR_MAX - DR_MIN)
    dc = (c1 - c0) / max(GRID_W-1, 1) * (DC_MAX - DC_MIN)
    total = np.sqrt(dr**2 + dc**2)
    return np.linspace(0, total, n)

# ==========================================

# BUILD FIGURE

# ==========================================

fig = plt.figure(figsize=(18, 10), facecolor='#ffffff')
fig.canvas.manager.set_window_title(f'Thermal Viewer — {os.path.basename(CSV_PATH)}')

gs = gridspec.GridSpec(2, 2,
    width_ratios=[1.3, 1],
    height_ratios=[1, 0.06],
    hspace=0.30, wspace=0.36,
    left=0.11, right=0.97, top=0.86, bottom=0.06)

ax_heat   = fig.add_subplot(gs[0, 0])     # heatmap
ax_prof   = fig.add_subplot(gs[0, 1])     # line / column profile
ax_slider = fig.add_subplot(gs[1, :])     # frame slider

for ax in (ax_heat, ax_prof):
    ax.set_facecolor('#ffffff')
    ax.tick_params(colors='#333333', labelsize=FONTSIZE)
    for sp in ax.spines.values(): sp.set_edgecolor('#cccccc')

# — Heatmap —

im = ax_heat.imshow(frames[0], cmap=CMAP, vmin=vmin, vmax=vmax,
    interpolation=INTERP, aspect='equal', origin='upper',
    extent=[DC_MIN - 0.5, DC_MAX + 0.5,
            DR_MAX + 0.5, DR_MIN - 0.5])
cb = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02)
cb.set_label('°C', color='#333333', fontsize=FONTSIZE)
cb.ax.yaxis.set_tick_params(color='#333333', labelcolor='#333333', labelsize=FONTSIZE)
ax_heat.set_xlabel('dc  (←  |  →)',             color='#555555', fontsize=FONTSIZE)
ax_heat.set_ylabel('dr  (↑ above  |  below ↓)', color='#555555', fontsize=FONTSIZE)
ax_heat.axhline(0, color='#cc9900', lw=0.7, linestyle='--', alpha=0.5)
ax_heat.axvline(0, color='#cc9900', lw=0.7, linestyle='--', alpha=0.5)

# Hover crosshair

hcross_h = ax_heat.axhline(0, color='#0077cc', lw=0.8, linestyle='-', alpha=0.7, visible=False)
hcross_v = ax_heat.axvline(0, color='#0077cc', lw=0.8, linestyle='-', alpha=0.7, visible=False)

# Hover temperature label

hover_text = ax_heat.text(0.02, 0.02, '', transform=ax_heat.transAxes,
    color='#0077cc', fontsize=FONTSIZE, va='bottom',
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='#0077cc', alpha=0.80))

# Frame / time label

frame_text = ax_heat.text(0.02, 0.97, '', transform=ax_heat.transAxes,
    color='#333333', fontsize=FONTSIZE, va='top',
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='#cccccc', alpha=0.85))

# Line overlay artists

line_overlay, = ax_heat.plot([], [], color='#ff4444', lw=1.5, zorder=5)
line_p0_dot,  = ax_heat.plot([], [], 'o', color='#ff4444', ms=6, zorder=6)
line_p1_dot,  = ax_heat.plot([], [], 'o', color='#ff4444', ms=6, zorder=6)
line_stats_text = ax_heat.text(0.98, 0.02, '', transform=ax_heat.transAxes,
    color='#cc0000', fontsize=FONTSIZE, va='bottom', ha='right',
    bbox=dict(boxstyle='round,pad=0.3', fc='#ffffff', ec='#cc0000', alpha=0.85))

# — Profile panel —

prof_line, = ax_prof.plot([], [], color='#cc0000', lw=2.5)
prof_fill  = None   # filled area between min/max across frames
prof_mean_line, = ax_prof.plot([], [], color='#888888', lw=1.5,
    linestyle='--', alpha=0.5, label='All-frame mean')
ax_prof.set_xlabel('Distance along line', color='#555555', fontsize=FONTSIZE)
ax_prof.set_ylabel('Temperature (°C)',    color='#555555', fontsize=FONTSIZE)
ax_prof.set_title('Line / column profile', color='#333333', fontsize=FONTSIZE*1.2, pad=4)
ax_prof.legend(facecolor='#ffffff', labelcolor='#333333', fontsize=FONTSIZE)
ax_prof.set_xlim(0, 1); ax_prof.set_ylim(vmin - 1, vmax + 1)

# — Slider —

ax_slider.set_facecolor("#f0f0f0")
slider = Slider(ax_slider, 'Frame', 0, N - 1, valinit=0, valstep=1,
    color='#2299aa', initcolor='none')
slider.label.set_color('#333333');  slider.label.set_fontsize(FONTSIZE)
slider.valtext.set_color('#333333'); slider.valtext.set_fontsize(FONTSIZE)

fig.suptitle(os.path.basename(CSV_PATH), color='#333333', fontsize=FONTSIZE, y=0.99)

# ==========================================

# STATE

# ==========================================

state = {
    'fi':         0,          # current frame index
    'hover_col':  GRID_W//2,  # hovered grid column
    'hover_row':  GRID_H//2,  # hovered grid row
    'line_p0':    None,       # (ax_x, ax_y) line start
    'line_p1':    None,       # (ax_x, ax_y) line end
    'drawing':    False,      # currently dragging
    'has_line':   False,      # a committed line exists
    'playing':    False,
    'timer':      None,
    'pixel_ts':   None,       # time-series for current hover/line
    'updating':   False,      # guard against slider re-entry
}

# ==========================================

# UPDATE FUNCTIONS

# ==========================================

def update_frame(fi):
    fi = int(np.clip(fi, 0, N-1))
    state['fi'] = fi
    grid = frames[fi]
    im.set_data(grid)
    frame_text.set_text(f'Frame {fi+1}/{N}  |  t = {timestamps[fi]:.1f} s')
    # Update hover display
    _refresh_hover(grid)
    # Update line profile
    if state['has_line']:
        _refresh_line_profile(grid)
    state['updating'] = True
    slider.set_val(fi)
    state['updating'] = False
    fig.canvas.draw_idle()

def _refresh_hover(grid):
    col, row = state['hover_col'], state['hover_row']
    if not state['has_line']:
        # Show column profile (all dr at this dc column)
        col_temps = grid[:, col]
        _update_profile_column(col_temps)
        # Time-series of this pixel
        pixel_ts = np.array([frames[i][row, col] for i in range(N)])
        state['pixel_ts'] = pixel_ts
    # Hover annotation
    temp = grid[row, col]
    ax_x = dc_axis[col] if col < len(dc_axis) else col
    ax_y = dr_axis[row] if row < len(dr_axis) else row
    hcross_h.set_ydata([ax_y, ax_y]); hcross_h.set_visible(True)
    hcross_v.set_xdata([ax_x, ax_x]); hcross_v.set_visible(True)
    hover_text.set_text(
        f'dc={dc_vals[col]:+d}  dr={dr_vals[row]:+d}\n{temp:.1f} °C')

def _update_profile_column(col_temps):
    """Show vertical (dr) profile for the hovered column."""
    x = dr_axis
    prof_line.set_data(col_temps, x)
    # All-frame mean for this column
    col_mean_all = np.array([frames[i][:, state['hover_col']] for i in range(N)])
    mean_all     = np.nanmean(col_mean_all, axis=0)
    prof_mean_line.set_data(mean_all, x)
    ax_prof.set_xlim(vmin - 1, vmax + 1)
    ax_prof.set_ylim(DR_MAX + 0.5, DR_MIN - 0.5)
    ax_prof.set_xlabel('Temperature (°C)', color='#555555', fontsize=FONTSIZE)
    ax_prof.set_ylabel('dr',               color='#555555', fontsize=FONTSIZE)
    ax_prof.set_title(f'Column profile  dc={dc_vals[state["hover_col"]]:+d}',
        color='#333333', fontsize=FONTSIZE, pad=4)

def _refresh_line_profile(grid):
    p0 = state['line_p0']; p1 = state['line_p1']
    c0, r0 = axes_to_grid(p0[0], p0[1])
    c1, r1 = axes_to_grid(p1[0], p1[1])
    temps_along = sample_line(grid, r0, c0, r1, c1)
    dists       = line_distances(r0, c0, r1, c1)
    prof_line.set_data(dists, temps_along)

    # All-frame mean and envelope along this line
    all_along = np.array([sample_line(frames[i], r0, c0, r1, c1) for i in range(N)])
    mean_along = np.nanmean(all_along, axis=0)
    prof_mean_line.set_data(dists, mean_along)

    ax_prof.set_xlim(dists[0], max(dists[-1], 0.01))
    ax_prof.set_ylim(vmin - 1, vmax + 1)
    ax_prof.set_xlabel('Distance (dr/dc units)', color='#555555', fontsize=FONTSIZE)
    ax_prof.set_ylabel('Temperature (°C)',       color='#555555', fontsize=FONTSIZE)
    ax_prof.set_title('Line profile', color='#333333', fontsize=FONTSIZE, pad=4)

    # Stats annotation
    t_mean = float(np.nanmean(temps_along))
    t_min  = float(np.nanmin(temps_along))
    t_max  = float(np.nanmax(temps_along))
    line_stats_text.set_text(
        f'mean {t_mean:.1f}  min {t_min:.1f}  max {t_max:.1f} °C')

    # Time-series of line mean
    line_ts = np.array([np.nanmean(sample_line(frames[i], r0, c0, r1, c1))
                        for i in range(N)])
    state['pixel_ts'] = line_ts

def clear_line():
    state['line_p0'] = None; state['line_p1'] = None
    state['has_line'] = False; state['drawing'] = False
    line_overlay.set_data([], [])
    line_p0_dot.set_data([], [])
    line_p1_dot.set_data([], [])
    line_stats_text.set_text('')
    _refresh_hover(frames[state['fi']])
    fig.canvas.draw_idle()

# ==========================================

# EVENT HANDLERS

# ==========================================

def on_motion(event):
    if event.inaxes != ax_heat: return
    ax_x, ax_y = event.xdata, event.ydata
    if ax_x is None or ax_y is None: return
    col, row = axes_to_grid(ax_x, ax_y)
    state['hover_col'] = col; state['hover_row'] = row

    if state['drawing'] and state['line_p0'] is not None:
        # Live line preview while dragging
        p0 = state['line_p0']
        line_overlay.set_data([p0[0], ax_x], [p0[1], ax_y])
        line_p1_dot.set_data([ax_x], [ax_y])

    _refresh_hover(frames[state['fi']])
    if state['has_line']:
        _refresh_line_profile(frames[state['fi']])
    fig.canvas.draw_idle()

def on_press(event):
    if event.inaxes != ax_heat: return
    if event.button == 3:          # right-click → clear
        clear_line(); return
    if event.button == 1:
        state['drawing'] = True
        state['has_line'] = False
        state['line_p0'] = (event.xdata, event.ydata)
        state['line_p1'] = None
        line_p0_dot.set_data([event.xdata], [event.ydata])
        line_p1_dot.set_data([], [])
        line_overlay.set_data([], [])
        line_stats_text.set_text('')

def on_release(event):
    if event.inaxes != ax_heat or not state['drawing']: return
    if event.button == 1:
        state['drawing'] = False
        p0 = state['line_p0']
        p1 = (event.xdata, event.ydata)
        # Only commit if the line has some length
        dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            # Treat as click — clear line, just hover
            clear_line(); return
        state['line_p1'] = p1
        state['has_line'] = True
        line_overlay.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line_p0_dot.set_data([p0[0]], [p0[1]])
        line_p1_dot.set_data([p1[0]], [p1[1]])
        _refresh_line_profile(frames[state['fi']])
        fig.canvas.draw_idle()

def on_key(event):
    if event.key == 'q' or event.key == 'escape':
        plt.close(fig)
    elif event.key == ' ':
        toggle_play()
    elif event.key == 'right':
        stop_play(); update_frame(state['fi'] + 1)
    elif event.key == 'left':
        stop_play(); update_frame(state['fi'] - 1)
    elif event.key == 'c':
        clear_line()
    elif event.key == 's':
        save_frame()
    elif event.key == 'e':
        save_line_export()

def on_scroll(event):
    if event.inaxes == ax_heat:
        delta = -1 if event.button == 'up' else 1
        stop_play()
        update_frame(state['fi'] + delta)

def on_slider(val):
    if not state['playing'] and not state['updating']:
        update_frame(int(val))

def toggle_play():
    if state['playing']:
        stop_play()
    else:
        state['playing'] = True
        _schedule_next_frame()

def stop_play():
    state['playing'] = False
    if state['timer'] is not None:
        state['timer'].stop()
        state['timer'] = None

def _schedule_next_frame():
    if not state['playing']: return
    fi = state['fi'] + 1
    if fi >= N:
        stop_play(); return
    update_frame(fi)
    interval_ms = int(1000.0 / PLAYBACK_FPS)
    state['timer'] = fig.canvas.new_timer(interval=interval_ms)
    state['timer'].add_callback(_tick)
    state['timer'].single_shot = True
    state['timer'].start()

def _tick():
    _schedule_next_frame()

def save_frame():
    out = os.path.join(
        os.path.dirname(CSV_PATH),
        f'frame_{state["fi"]:05d}_t{timestamps[state["fi"]]:.1f}s.png')
    fig.savefig(out, dpi=300, facecolor='#111111')
    print(f'Saved → {out}')

EXPORT_PATH = os.path.join(os.path.dirname(CSV_PATH), 'line_export.xlsx')

def save_line_export():
    """Append one row of pixel temperatures (wide format) to line_export.xlsx.

    Each row: Material (blank) | Temperature (blank) | T_Point (blank) | 1 | 2 | … | N
    Matches the Temp_Measurements.xlsx layout — fill in the first 3 cells manually.
    """
    if not state['has_line']:
        print('No line drawn — draw a line first (click and drag), then press E.')
        return
    p0 = state['line_p0']; p1 = state['line_p1']
    c0, r0 = axes_to_grid(p0[0], p0[1])
    c1, r1 = axes_to_grid(p1[0], p1[1])
    temps_along = sample_line_pixels(frames[state['fi']], r0, c0, r1, c1)

    row = {'Material': '', 'Temperature': '', 'T_Point': ''}
    row.update({i + 1: float(t) for i, t in enumerate(temps_along)})
    df_new = pd.DataFrame([row])

    try:
        if os.path.exists(EXPORT_PATH):
            df_existing = pd.read_excel(EXPORT_PATH)
            # Normalise numeric column names (Excel round-trips 1 → 1.0 → must be int again)
            df_existing.columns = [
                int(c) if isinstance(c, (int, float)) and not isinstance(c, bool)
                else (int(float(c)) if isinstance(c, str) and c.replace('.', '', 1).isdigit()
                      else c)
                for c in df_existing.columns
            ]
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_excel(EXPORT_PATH, index=False)
        print(f'Appended row with {len(temps_along)} pixels → {EXPORT_PATH}')
    except Exception as e:
        print(f'ERROR saving line export: {e}')

# ==========================================

# CONNECT EVENTS

# ==========================================

fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_press_event',  on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event',     on_key)
fig.canvas.mpl_connect('scroll_event',        on_scroll)
slider.on_changed(on_slider)

# ==========================================

# INITIAL DRAW

# ==========================================

update_frame(0)
plt.show()
w