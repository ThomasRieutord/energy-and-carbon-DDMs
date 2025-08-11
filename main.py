"""Energy and carbon footprint considerations for data-driven weather forecasting models

Main program.
"""

import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

# PREPARATION
# ===========

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="main.py",
    description="Reproduce the figures and the calculation given in the article.",
    epilog="Example: python main.py --nruns 365 --nmembers 51 --savefig --figmt svg",
)
parser.add_argument(
    "--nruns",
    help="The number of runs to make in the use-case (one year of forecast twice a day: 2*365=730)",
    default="2*365",
)
parser.add_argument(
    "--nmembers",
    help="The size of the ensemble",
    default="51",
)
parser.add_argument(
    "--countries",
    help="Countries to put in the carbon footprint table (comma-separated 3-letter ISO codes. Ex: NOR,ITA,POL,USA,CHN)",
    default="NOR,ITA,POL,USA,CHN",
)
parser.add_argument(
    "--train-country",
    help="Last column of the carbon footprint table: country in which training is done",
    default="NOR",
)
parser.add_argument(
    "--infer-country",
    help="Last column of the carbon footprint table: country in which inference is done",
    default="POL",
)
parser.add_argument(
    "--savefig", help="Save the figures instead of plotting them", action="store_true"
)
parser.add_argument("--figdir", help="Output directory for figures", default="figures")
parser.add_argument("--figfmt", help="Format for the figures (png, svg)", default="png")
args = parser.parse_args()

n_runs = eval(args.nruns)
n_mb = eval(args.nmembers)

countries = args.countries.split(",")
train_country = args.train_country
infer_country = args.infer_country


# Loading data
# ------------
ai = utils.load("review")
hw = utils.load("hardware")
cipc = utils.get_carbon_intensities()  # Carbon intensity per country (kgCO2e/MWh)

tdp = []
for h in ai["hardware"]:
    if h in hw["hardware"].values:
        tdp.append(hw[hw["hardware"] == h]["tdp_watt"].item())
    else:
        tdp.append(np.nan)

ai["tdp"] = tdp

# Physics-based counterpart
# -------------------------
ifs_energy_consumption_kWh = (  # From ECMWF Newletter no 181 https://www.ecmwf.int/sites/default/files/elibrary/82024/81616-newsletter-no-181-autumn-2024.pdf
    96  # on 96 AMD Epyc Rome CPUs
    * 3600  # "takes about one hour to produce (excluding I/O)"
    * 225  # with TDP of 225
    * (10 / 15)  # ratio for 10-day forecast
    / 3.6e6  # Joules to kWh
)

# Use case: two forecast per day with 51 members, for 1 year
oneyearoper_ifs = utils.usecase_energy_consumption(
    0, ifs_energy_consumption_kWh, n_members=n_mb, n_runs=n_runs
)

tco2_ifs = {}
for c in countries:
    tco2_ifs[c] = (
        cipc[c]["carbon_intensity"] * n_mb * n_runs * ifs_energy_consumption_kWh * 1e-6
    )


# Uncertainty quantification
# --------------------------
training_energy_uq = utils.energy_consumption_uq(
    ai["training_time"] * ai["n_gpus_training"] / 3600, ai["tdp"] / 1000
)
inference_energy_uq = utils.energy_consumption_uq(
    ai["inference_time"] * ai["n_gpus_inference"] / 3600, ai["tdp"] / 1000
)
oneyearoper_uq = utils.usecase_energy_consumption(
    training_energy_uq, inference_energy_uq, n_members=n_mb, n_runs=n_runs
)

# Computing quantities
# --------------------
ai["training_time_day"] = ai["training_time"] / (24*3600)
ai["training_energy_kWh"] = (
    ai["training_time"] * ai["n_gpus_training"] * ai["tdp"] / 3.6e6
)
ai["inference_energy_kWh"] = (
    ai["inference_time"] * ai["n_gpus_inference"] * ai["tdp"] / 3.6e6
)
ai["inference_energy_Wh"] = ai["inference_energy_kWh"] * 1000
ai["nrjratio_inference_train"] = ai["training_energy_kWh"] / ai["inference_energy_kWh"]
ai["nrjratio_ifs_inference"] = ifs_energy_consumption_kWh / ai["inference_energy_kWh"]
ai["nrjratio_training_ifs"] = ai["training_energy_kWh"] / ifs_energy_consumption_kWh
# Use case: 2 forecasts per day with 51 members for 1 year, 1 training per year
ai["oneyearoper_inference_kWh"] = n_mb * n_runs * ai["inference_energy_kWh"]
ai["oneyearoper_kWh"] = ai["training_energy_kWh"] + ai["oneyearoper_inference_kWh"]
ai["payback_boundary"] = ai["training_energy_kWh"] / (
    ifs_energy_consumption_kWh - ai["inference_energy_kWh"]
)
ai["largest_ensemble"] = ai["nrjratio_ifs_inference"] * n_mb
ai["onyearoper_ifs_ratio"] = oneyearoper_ifs / ai["oneyearoper_kWh"]
ai["onyearoper_train_percent"] = 100 * ai["training_energy_kWh"] / ai["oneyearoper_kWh"]
ai["payback_days_51mb"] = ai["payback_boundary"] / 102
ai["payback_days_1000mb"] = ai["payback_boundary"] / 2000
for c in countries:
    ai[f"tco2_oneyearoper_{c}"] = (
        cipc[c]["carbon_intensity"] * ai["oneyearoper_kWh"] * 1e-6
    )

ai[f"tco2_train_{train_country}_infer_{infer_country}"] = (
    cipc[train_country]["carbon_intensity"] * ai["training_energy_kWh"] * 1e-6
    + cipc[infer_country]["carbon_intensity"] * ai["oneyearoper_inference_kWh"] * 1e-6
)

# Selecting models
# ----------------
idxs = np.logical_and(
    ~np.isnan(ai["oneyearoper_kWh"]),
    np.logical_and(ai["resolution_deg"] < 1, ai["purpose"] == "global NWP"),
)
idxs = idxs[idxs].index.values

out = ai.loc[idxs, :]


# RESULTS
# =======

print(f"Use-case: n_runs={n_runs}, n_mb={n_mb}")

print(
    f"Energy consumpution of the IFS for a 10-day 1-member forecast: {round(ifs_energy_consumption_kWh, 1)} kWh"
)

# Tables
# ------
print("\n  Table 1: INPUT DATA\n")
utils.print_table(out)

print("\n  Table 2: OUTPUT DATA\n")
utils.print_table(
    round(out, 1),
    cols=[
        "name",
        "inference_energy_Wh",
        "nrjratio_ifs_inference",
        "training_energy_kWh",
        "nrjratio_training_ifs",
        "payback_days_51mb",
        "largest_ensemble",
    ],
)

print("\n  Table 3: CARBON FOOTPRINTS\n")
utils.print_table(
    round(out, 1),
    cols=["name"]
    + [f"tco2_oneyearoper_{c}" for c in countries]
    + [f"tco2_train_{train_country}_infer_{infer_country}"],
)
print("&\t".join(["IFS"] + [str(round(tco2_ifs[c], 1)) for c in countries] + ["-"]))


# Figures
# ------

### Energy consumption for one year of operation

plt.figure(figsize=(8, 7))
plt.title("Energy consumption for one year of operation")
plt.ylabel("kWh")
plt.bar(out["name"], out["training_energy_kWh"], label="Training")
plt.bar(
    out["name"],
    out["oneyearoper_inference_kWh"],
    bottom=out["training_energy_kWh"],
    label="Inference",
)
for i in range(len(oneyearoper_uq)):
    plt.plot(out["name"], oneyearoper_uq[i][idxs], "ko", alpha=0.5)

for x, i in enumerate(idxs):
    plt.text(x, out["oneyearoper_kWh"][i], round(out["onyearoper_ifs_ratio"][i], 1))

plt.legend()
if args.savefig:
    figpath = os.path.join(
        args.figdir,
        f"oneyearoper_conso.{args.figfmt}",
    )
    plt.savefig(figpath)
    print("Figure saved:", figpath)


### Carbon intensities

shpfilename = shpreader.natural_earth(
    resolution="110m", category="cultural", name="admin_0_countries"
)
reader = shpreader.Reader(shpfilename)
countries = list(reader.records())
gem = utils.get_carbon_intensities()
cipc = {k: v["carbon_intensity"] for k, v in gem.items()}

geometries = []
carbon_intensity = []

for country in countries:
    geometries.append(country.geometry)
    try:
        carbon_intensity.append(cipc[country.attributes["ADM0_A3"]])
    except KeyError:
        carbon_intensity.append(np.nan)

fig = plt.figure()
ax = fig.add_subplot(projection=ccrs.Robinson())
art = ax.add_geometries(
    geometries,
    crs=ccrs.PlateCarree(),
    array=carbon_intensity,
    cmap="magma_r",
)
ax.add_feature(cfeature.COASTLINE)

cbar = fig.colorbar(art, orientation="horizontal")
cbar.set_label("Carbon intensity (kgCO2/MWh)")
fig.suptitle("Country carbon intensity of electricity production")
if args.savefig:
    figpath = os.path.join(
        args.figdir,
        f"carbon_intensities.{args.figfmt}",
    )
    plt.savefig(figpath)
    print("Figure saved:", figpath)

plt.show()
