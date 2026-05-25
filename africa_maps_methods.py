#!/usr/bin/env python3
"""
West & Central Africa – Heat-CVD Map Generator
================================================
Produces 7 SVG files (place input files on ~/Desktop before running):
  01_koppen_map.svg       – Köppen-Geiger climate classification
  01_koppen_legend.svg    – Köppen legend (only zones present in data)
  02_tmrel_map.svg        – TMREL (Temperature of Minimum Mortality) in greens
  02_tmrel_legend.svg     – TMREL colour bar
  03_paf_map.svg          – PAF Median in reds
  03_paf_legend.svg       – PAF colour bar
  00_scale_bar.svg        – Standalone 100 km scale bar

Dependencies: geopandas, pandas, numpy, matplotlib, requests, shapely
  pip install geopandas pandas numpy matplotlib requests shapely pyogrio
"""

import io
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
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
    "koppen_map":    os.path.join(DESKTOP, "01_koppen_map.svg"),
    "koppen_legend": os.path.join(DESKTOP, "01_koppen_legend.svg"),
    "tmrel_map":     os.path.join(DESKTOP, "02_tmrel_map.svg"),
    "tmrel_legend":  os.path.join(DESKTOP, "02_tmrel_legend.svg"),
    "paf_map":       os.path.join(DESKTOP, "03_paf_map.svg"),
    "paf_legend":    os.path.join(DESKTOP, "03_paf_legend.svg"),
    "scale_bar":     os.path.join(DESKTOP, "00_scale_bar.svg"),
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

# ISO-3 codes for the West + Central Africa study countries (for clipping and borders)
STUDY_ISO3 = {
    "AGO","BEN","BFA","CMR","CAF","TCD","COD","COG","GNQ",
    "GAB","GMB","GHA","GIN","GNB","LBR","MLI","MRT","NER",
    "NGA","SEN","SLE","TGO","CIV",
}

# Map styling
OCEAN_COL    = "white"      # figure/background colour (no blue ocean)
AFRICA_GREY  = "#BEBEBE"    # non-study Africa fill
AFRICA_EDGE  = "#888888"    # African country border colour
STUDY_EDGE   = "#222222"    # study-region outer outline
COUNTRY_EDGE = "#444444"    # country borders inside the study region

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

# Study countries – used for clipping grid cells and drawing country borders
study_countries_gdf = africa[africa["ISO3166-1-Alpha-3"].isin(STUDY_ISO3)].copy()
study_union = study_countries_gdf.geometry.union_all()

# ── Clip grid cells to country boundaries for smooth edges ────────────────
print("Clipping grid cells to country boundaries…")
gdf["geometry"] = gdf.geometry.intersection(study_union)
gdf = gdf[~gdf.geometry.is_empty].copy()
print(f"  {len(gdf):,} grid cells after clipping")

# Dissolved outer boundary for the final outline ring
study_boundary = gdf.dissolve()

# Map extent: full Africa with a small margin so the continent is always visible
africa_bounds = africa.total_bounds   # [minx, miny, maxx, maxy]
pad = 2.5
XLIM = (africa_bounds[0] - pad, africa_bounds[2] + pad)
YLIM = (africa_bounds[1] - pad, africa_bounds[3] + pad)

# ──────────────────────────────────────────────────────────────────────────
# 4. HELPER: base map drawing
# ──────────────────────────────────────────────────────────────────────────

def make_map_figure():
    """Return (fig, ax) showing the full African continent."""
    fig, ax = plt.subplots(figsize=(13, 15))
    ax.set_facecolor(OCEAN_COL)
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
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
    """Country borders within the study region, drawn above data (zorder 9)."""
    study_countries_gdf.boundary.plot(
        ax=ax,
        color=COUNTRY_EDGE,
        linewidth=0.5,
        zorder=9,
    )


def draw_study_outline(ax):
    """Outer boundary of the study region (zorder 10)."""
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


# ──────────────────────────────────────────────────────────────────────────
# 5. MAP 1 – Köppen-Geiger
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering Köppen-Geiger map…")

present_zones = set(gdf["koppen_zone"].dropna().unique())
ordered_zones = [z for z in ZONE_ORDER if z in present_zones]

fig, ax = make_map_figure()
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
leg = ax.legend(
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
# 6. MAP 2 – TMREL  (shades of green, percentile-capped scale)
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering TMREL map…")

tmrel_vals = gdf["TMREL"].dropna().values
tmrel_p10  = float(np.percentile(tmrel_vals, 10))
tmrel_p90  = float(np.percentile(tmrel_vals, 90))
norm_tmrel = Normalize(vmin=tmrel_p10, vmax=tmrel_p90)
cmap_tmrel = cm.Greens

fig, ax = make_map_figure()
draw_africa(ax)
gdf.plot(
    ax=ax,
    column="TMREL",
    cmap=cmap_tmrel,
    norm=norm_tmrel,
    edgecolor="none",
    linewidth=0,
    zorder=2,
    missing_kwds={"color": "lightgrey", "edgecolor": "none"},
)
draw_country_borders(ax)
draw_study_outline(ax)
save_map(fig, OUTPUTS["tmrel_map"])

# ── Legend 2 ──────────────────────────────────────────────────────────────
print("Rendering TMREL legend…")

fig = plt.figure(figsize=(5.5, 1.5))
ax_cb = fig.add_axes([0.06, 0.55, 0.88, 0.28])
sm = cm.ScalarMappable(cmap=cmap_tmrel, norm=norm_tmrel)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
cbar.set_label(
    "Temperature of Minimum Relative Risk (TMREL, °C)",
    fontsize=10, labelpad=5
)
cbar.ax.tick_params(labelsize=9)

# 6 evenly-spaced ticks; outer labels indicate the percentile cap
ticks  = np.linspace(tmrel_p10, tmrel_p90, 6)
labels = [f"{t:.1f}" for t in ticks]
labels[0]  = f"<{tmrel_p10:.1f}"
labels[-1] = f">{tmrel_p90:.1f}"
cbar.set_ticks(ticks)
cbar.set_ticklabels(labels)

fig.savefig(OUTPUTS["tmrel_legend"], format="svg", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"  ✓  {OUTPUTS['tmrel_legend']}")

# ──────────────────────────────────────────────────────────────────────────
# 7. MAP 3 – PAF Median  (light → dark red, percentile-capped scale)
# ──────────────────────────────────────────────────────────────────────────
print("\nRendering PAF Median map…")

paf_vals = gdf["paf_median"].dropna().values
paf_p10  = float(np.percentile(paf_vals, 10))
paf_p90  = float(np.percentile(paf_vals, 90))
norm_paf = Normalize(vmin=paf_p10, vmax=paf_p90)
cmap_paf = LinearSegmentedColormap.from_list(
    "light_dark_red", ["#FFCCCC", "#CC0000", "#8B0000"], N=256
)

fig, ax = make_map_figure()
draw_africa(ax)
gdf.plot(
    ax=ax,
    column="paf_median",
    cmap=cmap_paf,
    norm=norm_paf,
    edgecolor="none",
    linewidth=0,
    zorder=2,
    missing_kwds={"color": "lightgrey", "edgecolor": "none"},
)
draw_country_borders(ax)
draw_study_outline(ax)
save_map(fig, OUTPUTS["paf_map"])

# ── Legend 3 ──────────────────────────────────────────────────────────────
print("Rendering PAF legend…")

paf_pct_p10  = paf_p10 * 100
paf_pct_p90  = paf_p90 * 100
norm_paf_pct = Normalize(vmin=paf_pct_p10, vmax=paf_pct_p90)

fig = plt.figure(figsize=(5.5, 1.5))
ax_cb = fig.add_axes([0.06, 0.55, 0.88, 0.28])
sm = cm.ScalarMappable(cmap=cmap_paf, norm=norm_paf_pct)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cb, orientation="horizontal")
cbar.set_label(
    "Population Attributable Fraction – median (PAF, %)",
    fontsize=10, labelpad=5
)
cbar.ax.tick_params(labelsize=9)

# 6 evenly-spaced ticks; outer labels indicate the percentile cap
ticks_pct  = np.linspace(paf_pct_p10, paf_pct_p90, 6)
labels_pct = [f"{t:.1f}%" for t in ticks_pct]
labels_pct[0]  = f"<{paf_pct_p10:.1f}%"
labels_pct[-1] = f">{paf_pct_p90:.1f}%"
cbar.set_ticks(ticks_pct)
cbar.set_ticklabels(labels_pct)

fig.savefig(OUTPUTS["paf_legend"], format="svg", bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"  ✓  {OUTPUTS['paf_legend']}")

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

  <!-- End ticks (vertical lines outside the bar) -->
  <line x1="30"  y1="15" x2="30"  y2="36"
        stroke="#222222" stroke-width="1.8"/>
  <line x1="120" y1="15" x2="120" y2="36"
        stroke="#222222" stroke-width="1.8"/>
  <line x1="210" y1="15" x2="210" y2="36"
        stroke="#222222" stroke-width="1.8"/>

  <!-- Tick labels above the bar -->
  <text x="30"  y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-size="10" fill="#222222">0</text>
  <text x="120" y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-size="10" fill="#222222">50</text>
  <text x="210" y="13" text-anchor="middle"
        font-family="Arial, Helvetica, sans-serif"
        font-size="10" fill="#222222">100</text>

  <!-- Unit label below the bar -->
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
║  All 7 SVG files saved to:  ~/Desktop/                  ║
╠══════════════════════════════════════════════════════════╣
║  00_scale_bar.svg       – 100 km scale bar              ║
║  01_koppen_map.svg      – Köppen-Geiger climate map     ║
║  01_koppen_legend.svg   – Köppen legend                 ║
║  02_tmrel_map.svg       – TMREL map (greens)            ║
║  02_tmrel_legend.svg    – TMREL colour bar              ║
║  03_paf_map.svg         – PAF Median map (reds)         ║
║  03_paf_legend.svg      – PAF colour bar                ║
╚══════════════════════════════════════════════════════════╝
""")
