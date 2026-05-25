#!/usr/bin/env python3
"""
West & Central Africa – Heat-CVD Map Generator
================================================
Produces 9 SVG files (place input files on ~/Desktop before running):
  01_koppen_map.svg        – Köppen-Geiger climate classification (full Africa)
  01_koppen_legend.svg     – Köppen legend (only zones present in data)
  02_tmrel_map.svg         – TMREL map, study-region zoom, custom greens
  02_tmrel_legend_h.svg    – TMREL horizontal colour bar
  02_tmrel_legend_v.svg    – TMREL vertical colour bar
  03_paf_map.svg           – PAF Median map, study-region zoom, custom reds
  03_paf_legend_h.svg      – PAF horizontal colour bar
  03_paf_legend_v.svg      – PAF vertical colour bar
  00_scale_bar.svg         – Standalone 100 km scale bar

Dependencies: geopandas, pandas, numpy, scipy, matplotlib, requests, shapely
  pip install geopandas pandas numpy scipy matplotlib requests shapely pyogrio
"""

import io
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap
import requests

# ──────────────────────────────────────────────────────────────────────────
# 0. PATHS
# ──────────────────────────────────────────────────────────────────────────
DESKTOP = Path('/Users/gokulparameswaran/Desktop')

GEOJSON_PATH = os.path.join(DESKTOP, "West_Central_Main.geojson")
CSV_PATH     = os.path.join(DESKTOP, "WCAfrica_Grid_HeatCVD_AgeStratified_Per100k.csv")

OUTPUTS = {
    "koppen_map":     os.path.join(DESKTOP, "01_koppen_map.svg"),
    "koppen_legend":  os.path.join(DESKTOP, "01_koppen_legend.svg"),
    "tmrel_map":      os.path.join(DESKTOP, "02_tmrel_map.svg"),
    "tmrel_legend_h": os.path.join(DESKTOP, "02_tmrel_legend_h.svg"),
    "tmrel_legend_v": os.path.join(DESKTOP, "02_tmrel_legend_v.svg"),
    "paf_map":        os.path.join(DESKTOP, "03_paf_map.svg"),
    "paf_legend_h":   os.path.join(DESKTOP, "03_paf_legend_h.svg"),
    "paf_legend_v":   os.path.join(DESKTOP, "03_paf_legend_v.svg"),
    "scale_bar":      os.path.join(DESKTOP, "00_scale_bar.svg"),
}

# ──────────────────────────────────────────────────────────────────────────
# 1. COLOURS & LABELS
# ──────────────────────────────────────────────────────────────────────────

# Standard Beck et al. 2018 Köppen-Geiger palette
KOPPEN_COLOURS = {
    "Af":  "#0000FF",
    "Am":  "#0078C8",
    "Aw":  "#96C8FF",
    "BWh": "#FF0000",
    "BWk": "#FFCCBC",
    "BSh": "#F0A000",
    "BSk": "#FFE09A",
    "Csa": "#FFFF00",
    "Csb": "#C8C800",
    "Csc": "#969600",
    "Cwa": "#96FF96",
    "Cwb": "#64C864",
    "Cwc": "#329632",
    "Cfa": "#C8FF50",
    "Cfb": "#64FF00",
    "Cfc": "#32C800",
    "Dsa": "#FF0078",
    "Dsb": "#C80064",
    "Dsc": "#960050",
    "Dsd": "#640032",
    "Dwa": "#C8B4FF",
    "Dwb": "#9664FF",
    "Dwc": "#6432C8",
    "Dwd": "#320096",
    "Dfa": "#C8C8FF",
    "Dfb": "#9696FF",
    "Dfc": "#6464C8",
    "Dfd": "#323296",
    "ET":  "#B2B2B2",
    "EF":  "#666666",
}

KOPPEN_LABELS = {
    "Af":  "Equatorial climate (Af)",
    "Am":  "Monsoon climate (Am)",
    "Aw":  "Tropical savanna climate (Aw)",
    "BWh": "Warm desert climate (BWh)",
    "BWk": "Cold desert climate (BWk)",
    "BSh": "Warm semi-arid climate (BSh)",
    "BSk": "Cold semi-arid climate (BSk)",
    "Csa": "Warm mediterranean climate (Csa)",
    "Csb": "Temperate mediterranean climate (Csb)",
    "Csc": "Cold mediterranean climate (Csc)",
    "Cwa": "Humid subtropical climate (Cwa)",
    "Cwb": "Subtropical oceanic highland climate (Cwb)",
    "Cwc": "Humid oceanic subarctic climate (Cwc)",
    "Cfa": "Warm oceanic / Humid subtropical climate (Cfa)",
    "Cfb": "Temperate oceanic climate (Cfb)",
    "Cfc": "Subpolar oceanic climate (Cfc)",
    "Dsa": "Hot-summer humid continental (Dsa)",
    "Dsb": "Warm-summer humid continental (Dsb)",
    "Dsc": "Subarctic climate (Dsc)",
    "Dsd": "Subarctic with severe winters (Dsd)",
    "Dwa": "Hot-summer humid continental (Dwa)",
    "Dwb": "Warm-summer humid continental (Dwb)",
    "Dwc": "Subarctic climate (Dwc)",
    "Dwd": "Subarctic with severe winters (Dwd)",
    "Dfa": "Hot-summer humid continental (Dfa)",
    "Dfb": "Warm-summer humid continental (Dfb)",
    "Dfc": "Subarctic climate (Dfc)",
    "Dfd": "Subarctic with severe winters (Dfd)",
    "ET":  "Tundra climate (ET)",
    "EF":  "Ice cap climate (EF)",
}

# Canonical zone order (for legend ordering)
ZONE_ORDER = [
    "Af", "Am", "Aw",
    "BWh", "BWk", "BSh", "BSk",
    "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc", "Cfa", "Cfb", "Cfc",
    "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd",
    "Dfa", "Dfb", "Dfc", "Dfd",
    "ET",  "EF",
]

# African ISO-3 codes
AFRICA_ISO3 = {
    "DZA","AGO","BEN","BWA","BFA","BDI","CMR","CPV","CAF","TCD",
    "COM","COD","COG","DJI","EGY","GNQ","ERI","SWZ","ETH","GAB","GMB",
    "GHA","GIN","GNB","KEN","LSO","LBR","LBY","MDG","MWI","MLI","MRT",
    "MUS","MYT","MAR","MOZ","NAM","NER","NGA","REU","RWA","STP","SEN",
    "SLE","SOM","ZAF","SSD","SDN","TZA","TGO","TUN","UGA","ESH","ZMB","ZWE",
    "CIV","SHN",
}

# ISO-3 codes for the West + Central Africa study countries
STUDY_ISO3 = {
    "AGO","BEN","BFA","CMR","CAF","TCD","COD","COG","GNQ",
    "GAB","GMB","GHA","GIN","GNB","LBR","MLI","MRT","NER",
    "NGA","SEN","SLE","TGO","CIV",
}

# Map styling
OCEAN_COL    = "white"      # figure background
AFRICA_GREY  = "#BEBEBE"    # non-study Africa fill
AFRICA_EDGE  = "#888888"    # African country border colour
STUDY_EDGE   = "#222222"    # study-region outer outline
COUNTRY_EDGE = "#444444"    # country borders inside study region

# Custom colormaps – both start from a clearly visible (not near-white) shade
# TMREL: visible medium-light green → dark forest green
CMAP_TMREL = LinearSegmentedColormap.from_list(
    "tmrel_green", ["#A8D5A8", "#388E3C", "#1B5E20"], N=256
)
# PAF: visible medium-light salmon-red → dark maroon
CMAP_PAF = LinearSegmentedColormap.from_list(
    "paf_red", ["#FF9999", "#CC0000", "#8B0000"], N=256
)

# ──────────────────────────────────────────────────────────────────────────
# 2. LOAD & MERGE DATA
# ──────────────────────────────────────────────────────────────────────────
print("Loading GeoJSON…")
gdf = gpd.read_file(GEOJSON_PATH)
gdf = gdf.set_crs("EPSG:4326", allow_override=True)

print("Loading CSV…")
df = pd.read_csv(CSV_PATH)

# Harmonise grid_id type for the merge
gdf["grid_id"] = gdf["grid_id"].astype(float)
df["grid_id"]  = df["grid_id"].astype(float)

gdf = gdf.merge(
    df[["grid_id", "koppen_zone", "TMREL", "paf_median"]],
    on="grid_id", how="left"
)

# Capture the full GeoJSON extent NOW – this is the authoritative study-region
# bounding box and must be saved before any geometry operations alter the data.
raw_bounds = gdf.total_bounds   # [minx, miny, maxx, maxy]

# ──────────────────────────────────────────────────────────────────────────
# 3. LOAD AFRICA BACKGROUND
# ──────────────────────────────────────────────────────────────────────────
print("Fetching Africa boundary…")
_url = (
    "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
)
resp = requests.get(_url, verify=False, timeout=60)
resp.raise_for_status()
world = gpd.read_file(io.BytesIO(resp.content))
africa = world[world["ISO3166-1-Alpha-3"].isin(AFRICA_ISO3)].copy()
africa = africa.set_crs("EPSG:4326", allow_override=True)

# Identify study countries spatially – every Africa country that intersects the
# grid footprint is included.  This avoids ISO-3 code mismatches dropping entire
# countries from the clip polygon and leaving gaps in the rendered map.
grid_footprint      = gdf.dissolve().geometry.union_all()
study_countries_gdf = africa[africa.geometry.intersects(grid_footprint)].copy()
study_union         = study_countries_gdf.geometry.union_all()

# ── Clip grid cells to country boundaries for smooth edges ────────────────
print("Clipping grid cells to country boundaries…")
gdf["geometry"] = gdf.geometry.intersection(study_union)
gdf = gdf[~gdf.geometry.is_empty].copy()
print(f"  {len(gdf):,} grid cells after clipping")

# ── Nearest-neighbour fill for any missing attribute values ───────────────
def fill_nearest(gdf_in, col):
    """Replace NaN values in col with the value of the geometrically nearest cell."""
    out   = gdf_in.copy()
    valid = out[col].notna()
    if valid.all() or not valid.any():
        return out
    centroids  = out.geometry.centroid
    valid_xy   = np.column_stack([centroids[valid].x.values,   centroids[valid].y.values])
    missing_xy = np.column_stack([centroids[~valid].x.values,  centroids[~valid].y.values])
    _, nn_idx  = KDTree(valid_xy).query(missing_xy)
    out.loc[~valid, col] = out.loc[valid, col].values[nn_idx]
    return out

print("Filling missing grid-cell values with nearest-neighbour…")
gdf = fill_nearest(gdf, "TMREL")
gdf = fill_nearest(gdf, "paf_median")
gdf = fill_nearest(gdf, "koppen_zone")

# Dissolved outer boundary for the final outline ring
study_boundary = gdf.dissolve()

# ── Map extents ────────────────────────────────────────────────────────────
# Köppen: full African continent
africa_bounds = africa.total_bounds
pad_africa    = 2.5
XLIM = (africa_bounds[0] - pad_africa, africa_bounds[2] + pad_africa)
YLIM = (africa_bounds[1] - pad_africa, africa_bounds[3] + pad_africa)

# TMREL / PAF: derived from the raw GeoJSON bounds (saved before any clipping)
# so the view always covers the complete study region regardless of which
# countries the downloaded shapefile matched.
pad_study  = 0.8
STUDY_XLIM = (raw_bounds[0] - pad_study, raw_bounds[2] + pad_study)
STUDY_YLIM = (raw_bounds[1] - pad_study, raw_bounds[3] + pad_study)

# ──────────────────────────────────────────────────────────────────────────
# 4. HELPER: base map drawing
# ──────────────────────────────────────────────────────────────────────────

def make_map_figure(xlim=None, ylim=None, figsize=(13, 15)):
    """Return (fig, ax) for the given extent. Defaults to full-Africa view."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(OCEAN_COL)
    ax.set_xlim(xlim if xlim is not None else XLIM)
    ax.set_ylim(ylim if ylim is not None else YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def draw_africa(ax):
    """Grey Africa as base layer (zorder 1)."""
    africa.plot(
        ax=ax,
        color=AFRICA_GREY,
        edgecolor=AFRICA_EDGE,
        linewidth=0.25,
        zorder=1,
    )


def draw_country_borders(ax):
    """Country borders within the study region, above the data layer (zorder 9)."""
    study_countries_gdf.boundary.plot(
        ax=ax,
        color=COUNTRY_EDGE,
        linewidth=0.5,
        zorder=9,
    )


def draw_study_outline(ax):
    """Thick outer boundary of the study region (zorder 10)."""
    study_boundary.boundary.plot(
        ax=ax,
        color=STUDY_EDGE,
        linewidth=0.9,
        zorder=10,
    )


def save_map(fig, path):
    fig.savefig(path, format="svg", bbox_inches="tight",
                facecolor=OCEAN_COL, dpi=150)
    plt.close(fig)
    print(f"  ✓  {path}")


def make_framed_study_map():
    """Return (fig, ax) for the study region with a visible axes box,
    lat/lon tick labels, degree symbols, and grid — matching heat_cvd_maps.py."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(raw_bounds[0] - 0.5, raw_bounds[2] + 0.5)
    ax.set_ylim(raw_bounds[1] - 0.5, raw_bounds[3] + 0.5)
    ax.set_xlabel("Longitude", fontsize=12, fontweight="bold")
    ax.set_ylabel("Latitude",  fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, p: f"{x:.0f}°E" if x >= 0 else f"{abs(x):.0f}°W")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, p: f"{y:.0f}°N" if y >= 0 else f"{abs(y):.0f}°S")
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor("black")
    return fig, ax


def save_framed_map(fig, path):
    plt.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {path}")


def save_colorbar(sm, label, ticks, tick_labels, path_h, path_v):
    """Save both a horizontal and a vertical colourbar legend."""
    # ── Horizontal ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.0, 1.8))
    ax_cb = fig.add_axes([0.06, 0.55, 0.88, 0.28])
    cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cbar.set_label(label, fontsize=10, labelpad=5)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    fig.savefig(path_h, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {path_h}")

    # ── Vertical ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(2.6, 8.0))
    ax_cb = fig.add_axes([0.30, 0.06, 0.22, 0.88])
    cbar = fig.colorbar(sm, cax=ax_cb, orientation="vertical")
    cbar.set_label(label, fontsize=9, labelpad=10, rotation=270, va="bottom")
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(tick_labels)
    fig.savefig(path_v, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {path_v}")


# ──────────────────────────────────────────────────────────────────────────
# 5. MAP 1 – Köppen-Geiger  (full-Africa view)
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering Köppen-Geiger map…")

present_zones = set(gdf["koppen_zone"].dropna().unique())
ordered_zones = [z for z in ZONE_ORDER if z in present_zones]

fig, ax = make_map_figure()          # default = full Africa
draw_africa(ax)

for zone in ordered_zones:
    subset = gdf[gdf["koppen_zone"] == zone]
    if not subset.empty:
        subset.plot(
            ax=ax,
            color=KOPPEN_COLOURS[zone],
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )

draw_country_borders(ax)
draw_study_outline(ax)
save_map(fig, OUTPUTS["koppen_map"])

# ── Legend 1 ──────────────────────────────────────────────────────────────
print("Rendering Köppen legend…")

patches = [
    mpatches.Patch(
        facecolor=KOPPEN_COLOURS[z],
        edgecolor="#555555",
        linewidth=0.6,
        label=KOPPEN_LABELS.get(z, z),
    )
    for z in ordered_zones
]

n_patches = len(patches)
fig_h = n_patches * 0.42 + 0.8
fig, ax = plt.subplots(figsize=(5.8, fig_h))
ax.axis("off")
ax.legend(
    handles=patches,
    loc="center",
    frameon=True,
    framealpha=1.0,
    edgecolor="#AAAAAA",
    fontsize=10,
    title="Köppen-Geiger Climate Classification",
    title_fontsize=11,
    handlelength=1.8,
    handleheight=1.1,
    labelspacing=0.45,
)
fig.tight_layout(pad=0.3)
fig.savefig(OUTPUTS["koppen_legend"], format="svg", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"  ✓  {OUTPUTS['koppen_legend']}")

# ──────────────────────────────────────────────────────────────────────────
# 6. MAP 2 – TMREL  (study-region zoom, custom greens, percentile scale)
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering TMREL map…")

tmrel_vals = gdf["TMREL"].dropna().values
tmrel_p10  = float(np.percentile(tmrel_vals, 10))
tmrel_p90  = float(np.percentile(tmrel_vals, 90))
norm_tmrel = Normalize(vmin=tmrel_p10, vmax=tmrel_p90)

fig, ax = make_framed_study_map()
gdf.plot(
    ax=ax,
    column="TMREL",
    cmap=CMAP_TMREL,
    norm=norm_tmrel,
    linewidth=0.05,
    edgecolor="face",
    zorder=1,
)
study_countries_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor="#333333", zorder=2)
save_framed_map(fig, OUTPUTS["tmrel_map"])

# ── Legends 2 (H + V) ─────────────────────────────────────────────────────
print("Rendering TMREL legends…")

ticks_tmrel  = np.linspace(tmrel_p10, tmrel_p90, 6)
labels_tmrel = [f"{t:.1f}" for t in ticks_tmrel]
labels_tmrel[0]  = f"<{tmrel_p10:.1f}"
labels_tmrel[-1] = f">{tmrel_p90:.1f}"
sm_tmrel = cm.ScalarMappable(cmap=CMAP_TMREL, norm=norm_tmrel)
sm_tmrel.set_array([])

save_colorbar(
    sm_tmrel,
    label="Temperature of Minimum Relative Risk (TMREL, °C)",
    ticks=ticks_tmrel,
    tick_labels=labels_tmrel,
    path_h=OUTPUTS["tmrel_legend_h"],
    path_v=OUTPUTS["tmrel_legend_v"],
)

# ──────────────────────────────────────────────────────────────────────────
# 7. MAP 3 – PAF Median  (study-region zoom, custom reds, percentile scale)
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering PAF Median map…")

paf_vals = gdf["paf_median"].dropna().values
paf_p10  = float(np.percentile(paf_vals, 10))
paf_p90  = float(np.percentile(paf_vals, 90))
norm_paf = Normalize(vmin=paf_p10, vmax=paf_p90)

fig, ax = make_framed_study_map()
gdf.plot(
    ax=ax,
    column="paf_median",
    cmap=CMAP_PAF,
    norm=norm_paf,
    linewidth=0.05,
    edgecolor="face",
    zorder=1,
)
study_countries_gdf.boundary.plot(ax=ax, linewidth=0.8, edgecolor="#333333", zorder=2)
save_framed_map(fig, OUTPUTS["paf_map"])

# ── Legends 3 (H + V) ─────────────────────────────────────────────────────
print("Rendering PAF legends…")

paf_pct_p10  = paf_p10 * 100
paf_pct_p90  = paf_p90 * 100
norm_paf_pct = Normalize(vmin=paf_pct_p10, vmax=paf_pct_p90)

ticks_paf  = np.linspace(paf_pct_p10, paf_pct_p90, 6)
labels_paf = [f"{t:.1f}%" for t in ticks_paf]
labels_paf[0]  = f"<{paf_pct_p10:.1f}%"
labels_paf[-1] = f">{paf_pct_p90:.1f}%"
sm_paf = cm.ScalarMappable(cmap=CMAP_PAF, norm=norm_paf_pct)
sm_paf.set_array([])

save_colorbar(
    sm_paf,
    label="Population Attributable Fraction – median (PAF, %)",
    ticks=ticks_paf,
    tick_labels=labels_paf,
    path_h=OUTPUTS["paf_legend_h"],
    path_v=OUTPUTS["paf_legend_v"],
)

# ──────────────────────────────────────────────────────────────────────────
# 8. SCALE BAR – 100 km standalone SVG
# ──────────────────────────────────────────────────────────────────────────
print("\nGenerating scale bar…")

SCALEBAR_SVG = """\
<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="240" height="52" viewBox="0 0 240 52">

  <!-- White rounded background -->
  <rect x="0" y="0" width="240" height="52"
        fill="white" stroke="#CCCCCC" stroke-width="0.8"
        rx="5" ry="5"/>

  <!-- ── Bar segments (total = 180 px = 100 km) ── -->
  <!-- Left half: 0-50 km  black -->
  <rect x="30" y="20" width="90" height="12"
        fill="#222222"/>
  <!-- Right half: 50-100 km  white -->
  <rect x="120" y="20" width="90" height="12"
        fill="#ffffff" stroke="#222222" stroke-width="1"/>

  <!-- End ticks -->
  <line x1="30"  y1="15" x2="30"  y2="36" stroke="#222222" stroke-width="1.8"/>
  <line x1="120" y1="15" x2="120" y2="36" stroke="#222222" stroke-width="1.8"/>
  <line x1="210" y1="15" x2="210" y2="36" stroke="#222222" stroke-width="1.8"/>

  <!-- Tick labels above -->
  <text x="30"  y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#222222">0</text>
  <text x="120" y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#222222">50</text>
  <text x="210" y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif" font-size="10" fill="#222222">100</text>

  <!-- Unit label below -->
  <text x="120" y="48" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-size="11" font-weight="bold" fill="#222222">km</text>
</svg>
"""

with open(OUTPUTS["scale_bar"], "w", encoding="utf-8") as fh:
    fh.write(SCALEBAR_SVG)
print(f"  ✓  {OUTPUTS['scale_bar']}")

# ──────────────────────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════╗
║  All 9 SVG files saved to:  ~/Desktop/                  ║
╠══════════════════════════════════════════════════════════╣
║  00_scale_bar.svg        – 100 km scale bar             ║
║  01_koppen_map.svg       – Köppen-Geiger climate map    ║
║  01_koppen_legend.svg    – Köppen legend                ║
║  02_tmrel_map.svg        – TMREL map (greens, zoomed)   ║
║  02_tmrel_legend_h.svg   – TMREL horizontal colour bar  ║
║  02_tmrel_legend_v.svg   – TMREL vertical colour bar    ║
║  03_paf_map.svg          – PAF Median map (reds, zoomed)║
║  03_paf_legend_h.svg     – PAF horizontal colour bar    ║
║  03_paf_legend_v.svg     – PAF vertical colour bar      ║
╚══════════════════════════════════════════════════════════╝
""")
