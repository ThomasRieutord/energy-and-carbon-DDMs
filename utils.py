"""Energy and carbon footprint considerations for data-driven weather forecasting models

Utilities.
"""

import datetime as dt
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codecarbon.input import DataSource


def process_dates(df) -> pd.DataFrame:
    """Convert dates from string type ISO format (YYYY-MM-DD) to datetime.date and sort entries by date"""
    df["date"] = [dt.date.fromisoformat(d) for d in df["date"]]
    return df.sort_values("date")


def print_table(
    df,
    idxs=None,
    cols=[
        "name",
        "inference_time",
        "training_time_day",
        "n_gpus_training",
        "hardware",
        "tdp",
        "doi",
    ],
    style="latex",
) -> None:
    """Print the content of a dataframe as a LateX or Markdown table


    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe containing the data

    idxs: list or None
        Subset of index to print. If None (default), takes the full index

    cols: list of str
        Columns to display

    style: {"markdown", "latex"}, default = "latex"
        Style of the table (Markdown or LateX)
    """
    if style.lower() in ["md", "markdown"]:
        _print_markdown_table(df, idxs, cols)
    elif style.lower() in ["tex", "latex"]:
        _print_latex_table(df, idxs, cols)
    else:
        raise ValueError(
            f"Invalid value for style: {style}. Please choose among 'markdown', 'latex'"
        )


def _print_latex_table(df, idxs, cols) -> None:
    """Print the content of a dataframe as a Markdown table"""
    if idxs is None:
        idxs = df.index

    print("&\t".join(cols) + "\\\\")
    for i in idxs:
        print("&\t".join([str(df.loc[i, c]) for c in cols]) + "\\\\")


def _print_markdown_table(df, idxs, cols) -> None:
    """Print the content of a dataframe as a LateX table"""
    if idxs is None:
        idxs = df.index

    print("| " + " | ".join(cols) + " |")
    print("| :" + ": | :".join([(len(c) - 2) * "-" for c in cols]) + ": |")
    for i in idxs:
        print("| " + " | ".join([str(df.loc[i, c]) for c in cols]) + " |")


def load(what="review") -> pd.DataFrame:
    """Load data from CSV files with pre-set recipes.


    Parameters
    ----------
    what: {"review", "hardware"}
        The recipe to apply:
        "review" loads the data about the literature
        "hardware" loads the data about the hardware
        <path> loads the data from the given file


    Examples
    --------
    >>> ai = utils.load("review")
    >>> hw = utils.load("hardware")
    """
    if what in ["review", "ai-models-review.csv", "ai-models-review"]:
        path = os.path.join("data", "ai-models-review.csv")
        transform = process_dates
    elif what in ["hardware", "hardware-specs.csv", "hardware-specs"]:
        path = os.path.join("data", "hardware-specs.csv")
        transform = lambda x: x
    elif os.path.isfile(what):
        path = what
        transform = lambda x: x
    else:
        raise ValueError(
            f"Unable to load {what}. Please choose among ['review', 'hardware']"
        )

    skiprows = 0
    with open(path, "r") as f:
        for l in f:
            if l.startswith("#"):
                skiprows += 1
            else:
                break

    return transform(pd.read_csv(path, sep=",", skiprows=skiprows))


def get_high_mid_low(base, uncertainty, array=False):
    """Return a 3-tuple of high, medium, low values around a base value


    Parameters
    ----------
    base: float
        The base value

    uncertainty: array-like of length 3
        Relative proportion of the base value to be added for the high, medium and low values

    array: bool, optional
        If True, returns a Numpy array instead of a tuple


    Returns
    -------
    high: float
        High value = base*(1 + uncertainty[0])

    mid: float
        Medium value = base*(1 + uncertainty[1])

    low: float
        Low value = base*(1 + uncertainty[2])


    Example
    -------
    >>> utils.get_high_mid_low(100, [0.05 , 0, -0.15])
    (105.0, 100, 85.0)
    """
    high = base * (1 + uncertainty[0])
    mid = base * (1 + uncertainty[1])
    low = base * (1 + uncertainty[2])
    if array:
        return np.array([high, mid, low])
    else:
        return high, mid, low


def all_products(a, b):
    """Return a list of all item-wise products from two array-like


    Parameters
    ----------
    a: array-like
        First array

    b: array-like
        Second array


    Returns
    -------
    all_products: list of length len(a)*len(b)
        List of item-wise products


    Example
    -------
    >>> a = [3,3,3]
    >>> b = [1,2]
    >>> all_products(a,b)
    [3, 3, 3, 6, 6, 6]
    """
    return [a[i] * b[j] for j in range(len(b)) for i in range(len(a))]


def energy_consumption_uq(
    base_time,
    base_power,
    time_uncertainty=[0.05, 0, -0.05],
    power_uncertainty=[0, -0.25, -0.5],
) -> list:
    """Compute the energy consumption with uncertainty quantification

    The energy consumption is the product P*T between a power P and a
    time T. The values of P and T are taken from a base value modulated
    by an uncertainty quantification (high, medium and low values,
    see `get_high_mid_low`).


    Parameters
    ----------
    base_time: float
        The base value for the time (hours)

    base_power: float
        The base value for the power (kilowatts)

    time_uncertainty: array-like of length 3
        Uncertainty quantification on time

    power_uncertainty: array-like of length 3
        Uncertainty quantification on power


    Returns
    -------
    energy_consumption_uq: list of length 9
        Energy consumption estimated for each of the combination in the uncertainty ranges (kilowatt-hours)


    Example
    -------
    >>> # Energy consumed by a device draining 100W +/- 10% for 10 hours +/- 5%
    >>> # Approximately 0.1*10 = 1 kWh
    >>> utils.energy_consumption_uq(10, 0.1, time_uncertainty = [0.05, 0, -0.05], power_uncertainty = [0.1, 0, -0.1])
    [1.155, 1.1, 1.045, 1.05, 1.0, 0.95, 0.945, 0.9, 0.855]
    """
    return all_products(
        get_high_mid_low(base_time, time_uncertainty),
        get_high_mid_low(base_power, power_uncertainty),
    )


def usecase_energy_consumption(
    training_energy, inference_energy, n_members=51, n_runs=2 * 10 * 365
):
    """Compute the energy consumed by a DDM for a given use-case.

    The DDM consumes the energy `training_energy` for the training and the
    energy `inference_energy` for a single inference. The use-case is defined
    by the number of forecasts `n_runs` and the size of the ensemble `n_members`.


    Parameters
    ----------
    training_energy: float or array-like
        Energy consumed by the DDM(s) for the training

    inference_energy: float or array-like, same as `training_energy`
        Energy consumed by the DDM(s) for a single inference

    n_members: int
        The size of the ensemble made at each run

    n_runs: int
        The number of runs (1 run = 1 ensemble forecast)


    Returns
    -------
    total_energy: float or array-like, same as `training_energy`
        The total energy consumed for the use case
    """
    try:
        return [
            train + n_members * n_runs * infer
            for train, infer in zip(training_energy, inference_energy)
        ]
    except TypeError:
        return training_energy + n_members * n_runs * inference_energy


def get_carbon_intensities_iceland() -> dict:
    """Return carbon intensity for Iceland

    Data come from this [report](https://gogn.orkustofnun.is/Talnaefni/OS-2021-T014-01.pdf)
    of the National Energy Authority (NEA, Orkustofnun in Icelandic).
    Noticeably, it is not present in the CodeCarbon database (version 1.2.0).
    """
    ds = DataSource()
    cips = ds.get_carbon_intensity_per_source_data()  # kgCO2/Mwh
    iceland_data = (
        {  # Source: https://gogn.orkustofnun.is/Talnaefni/OS-2021-T014-01.pdf
            "ISL": {
                "biofuel_Twh": 0.0,
                "carbon_intensity": np.nan,
                "coal_Twh": 0.0,
                "country_name": "Iceland",
                "fossil_Twh": 30.7,
                "gaz_Twh": 0.0,
                "hydroelectricity_Twh": 13157.0,
                "iso_code": "ISL",
                "low_carbon_Twh": 5960.0,
                "nuclear_Twh": 0.0,
                "oil_Twh": 30.7,
                "other_renewable_Twh": 0.0,
                "other_renewable_exc_biofuel_Twh": 0.0,
                "per_capita_Twh": 0,
                "renewables_Twh": 16.33,
                "solar_Twh": 0.0,
                "total_Twh": 0,
                "wind_Twh": 6.66,
                "year": 2020,
            }
        }
    )
    iceland_total_Twh = sum(
        [v for k, v in iceland_data["ISL"].items() if k.endswith("Twh")]
    )
    iceland_carbon_intensity = sum(
        [
            (iceland_data["ISL"]["oil_Twh"] / iceland_total_Twh) * cips["petroleum"],
            (iceland_data["ISL"]["wind_Twh"] / iceland_total_Twh) * cips["wind"],
            (iceland_data["ISL"]["hydroelectricity_Twh"] / iceland_total_Twh)
            * cips["hydroelectricity"],
            (iceland_data["ISL"]["low_carbon_Twh"] / iceland_total_Twh)
            * cips["geothermal"],
        ]
    )
    iceland_data["ISL"]["total_Twh"] = iceland_total_Twh
    iceland_data["ISL"]["carbon_intensity"] = iceland_carbon_intensity

    return iceland_data


def get_carbon_intensities(include_iceland=True):
    """Return a dictionary with carbon intensities for each country.

    Keys of the dictionary are 3-letters ISO name of countries. Values are
    dictionaries with the energy mix data, including the carbon intensities,
    given in kgCO2/Mwh.


    Parameters
    ----------
    include_iceland: bool
        If True, data from Iceland is included


    Returns
    -------
    gem: dict
        Global energy mix data, including carbon intensities


    Examples
    --------
    >>> gem = get_carbon_intensities()
    # Italy: 226.196 kgCO2/Mwh
    >>> gem["ITA"]["carbon_intensity"]
    226.196
    """
    ds = DataSource()
    gem = ds.get_global_energy_mix_data()

    if include_iceland:
        iceland_data = get_carbon_intensities_iceland()
        gem.update(iceland_data)

    return gem
