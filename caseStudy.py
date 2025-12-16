import mainCode as withRandomness
import pandas as pd
import math
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from pprint import pprint


# function to turn temperatures into egg laying percentage
def get_percentage_from_temp(T, T_opt=25, sigma=8):
    return math.exp(-((T - T_opt)**2) / (2 * sigma**2))

def germany():
    # for germany
    data_germany = pd.read_csv("germany.csv")
    germany_size = 61

    data_germany = data_germany.set_index("day")

    # array for temperatures
    germany_percentages = []
    for temp in data_germany["avg_temp"]:
        germany_percentages.append(get_percentage_from_temp(temp))

    # if simulate expects up to 100 days, keep your padding
    # germany_percentages += [1] * (100 - germany_size)

    # simulate(Tmax, dt, initState, p, num_of_pher=5, temperatures=None):
    result = withRandomness.simulate(
        germany_size, 1,
        withRandomness.stateInit,
        withRandomness.params,
        30,
        germany_percentages
    )

    print("Germany result shape:", result.shape)
    print(result.head())

    # return population data and the avg_temp SERIES
    return result, data_germany["avg_temp"]


def brazil():
    df = pd.read_csv("brazilTemperatures.CSV", sep=";")
    df = df[["date", "hour", "hourly_temperatures"]]
    df["date"] = pd.to_datetime(df["date"])

    df_daily = (
        df.groupby("date")["hourly_temperatures"]
          .mean()
          .reset_index()
          .rename(columns={"hourly_temperatures": "avg_temp"})
    )

    df_sept_oct = df_daily[
        (df_daily["date"].dt.month == 9) |
        (df_daily["date"].dt.month == 10)
    ].reset_index(drop=True)

    df_sept_oct.insert(0, "day", range(1, len(df_sept_oct) + 1))
    df_final = df_sept_oct[["day", "avg_temp"]]
    
    # final data in correct form
    data_brazil = df_final.set_index("day")
    brazil_size = 61

    brazil_percentages = []
    for temp in data_brazil["avg_temp"]:
        brazil_percentages.append(get_percentage_from_temp(temp))

    result = withRandomness.simulate(
        brazil_size, 1,
        withRandomness.stateInit,
        withRandomness.params,
        30,
        brazil_percentages
    )

    print("Brazil result shape:", result.shape)
    print(result.head())

    return result, data_brazil["avg_temp"]



def south_africe():

    # get and procces data
    data_SA = pd.read_csv("data_SA.csv")
    data_southAfrice = data_SA[["date", "tavg"]]
    data_SA = pd.read_csv("data_SA.csv")

    data_southAfrice = data_SA[["date", "tavg"]].copy()
    data_southAfrice["day"] = range(1, len(data_southAfrice) + 1)
    data_southAfrice = data_southAfrice[["day", "tavg"]]
    data_southAfrice.rename(columns={"tavg": "avg_temp"}, inplace=True)
    print(data_southAfrice.head())
    data_southAfrice = data_southAfrice.set_index("day")

    SA_size = 61

    southAfrice_percentages = []
    for temp in data_southAfrice["avg_temp"]:
        southAfrice_percentages.append(get_percentage_from_temp(temp))

    result = withRandomness.simulate(
        SA_size, 1,
        withRandomness.stateInit,
        withRandomness.params,
        30,
        southAfrice_percentages
    )

    print("South Africa result shape:", result.shape)
    print(result.head())

    return result, data_southAfrice["avg_temp"]


def antarctica():
    # for germany
    data_antractica = pd.read_csv("antarctica_may_june_avg.csv")
    antarctice_size = 61

    data_antractica = data_antractica.set_index("day")

    # array for temperatures
    antarctica_percenatges = []
    for temp in data_antractica["avg_temp"]:
        antarctica_percenatges.append(get_percentage_from_temp(temp))

    # if simulate expects up to 100 days, keep your padding
    # germany_percentages += [1] * (100 - germany_size)

    # simulate(Tmax, dt, initState, p, num_of_pher=5, temperatures=None):
    result = withRandomness.simulate(
        antarctice_size, 1,
        withRandomness.stateInit,
        withRandomness.params,
        30,
        antarctica_percenatges
    )
    print("Antarctica result shape:", result.shape)
    print(result.head())

    # return population data and the avg_temp SERIES
    return result, data_antractica["avg_temp"]

def death_valley():
    # for germany
    data_dv = pd.read_csv("temps.csv")
    dv_size = 61

    data_dv = data_dv.set_index("day")

    # array for temperatures
    dv_percentages = []
    for temp in data_dv["avg_temp"]:
        dv_percentages.append(get_percentage_from_temp(temp))

    # if simulate expects up to 100 days, keep your padding
    # germany_percentages += [1] * (100 - germany_size)

    # simulate(Tmax, dt, initState, p, num_of_pher=5, temperatures=None):
    result = withRandomness.simulate(
        dv_size, 1,
        withRandomness.stateInit,
        withRandomness.params,
        30,
        dv_percentages
    )
    print("Death Valley result shape:", result.shape)
    print(result.head())

    # return population data and the avg_temp SERIES
    return result, data_dv["avg_temp"]



def plot_func_gradients(pop_de, temps_de,
              pop_br, temps_br,
              pop_sa, temps_sa,
              pop_ant, temps_ant,
              pop_dv, temps_dv):

    # Pack all series into a list to avoid repeating code
    series = [
        dict(name="Germany",      pop=pop_de, temps=temps_de, cmap="coolwarm"),
        dict(name="Brazil",       pop=pop_br, temps=temps_br, cmap="viridis"),
        dict(name="South Africa", pop=pop_sa, temps=temps_sa, cmap="plasma"),
        dict(name="Antarctica",        pop=pop_ant, temps=temps_ant, cmap="magma"),
        dict(name="Death Vallye",           pop=pop_dv, temps=temps_dv, cmap="cividis"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    all_y = []
    all_x_lengths = []
    colorbars = []
    legend_entries = []

    for s in series:
        # ---------- Extract data ----------
        y = s["pop"]["total"].to_numpy()
        x = np.arange(len(y))
        t = s["temps"].to_numpy()

        all_y.append(y)
        all_x_lengths.append(len(y))

        # ---------- Build segments ----------
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Center normalisation at mean temperature for that country
        vcenter = np.mean(t)
        norm = TwoSlopeNorm(vmin=t.min(), vcenter=vcenter, vmax=t.max())

        lc = LineCollection(
            segments,
            cmap=s["cmap"],
            norm=norm
        )
        lc.set_array(t[:-1])
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        colorbars.append((line, s["name"], norm))

        # For legend: average temperature color
        avg_temp = np.mean(t)
        avg_color = plt.get_cmap(s["cmap"])(norm(avg_temp))

        legend_entries.append(
            Line2D([0], [0], color=avg_color, lw=3,
                   label=f"{s['name']} (avg temp color)")
        )

    # ---------- AXES LIMITS ----------
    y_min = min(y.min() for y in all_y)
    y_max = max(y.max() for y in all_y)
    max_len = max(all_x_lengths)

    ax.set_xlim(0, max_len + 1)
    ax.set_ylim(y_min - 500, y_max + 500)

    # ---------- COLORBARS ----------
    """
    for line, name, _norm in colorbars:
        cbar = plt.colorbar(line, ax=ax)
        cbar.set_label(f"Temperature ({name})")
    """
    # ---------- LEGEND ----------
    # ax.legend(handles=legend_entries, title="Countries")

    # ---------- LABELS / TITLE ----------
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(
        "Bee Population Colored by Temperature\n"
        "Germany, Brazil, South Africa, "
        "Antarctice, Death-Valley (US)"
    )

    plt.show()



def plot_func_nums(pop_de, temps_de,
              pop_br, temps_br,
              pop_sa, temps_sa,
              pop_ant, temps_ant,
              pop_dv, temps_dv):

    # Pack all series into a list to avoid repeating code
    series = [
        dict(name="Germany",      pop=pop_de,  temps=temps_de),
        dict(name="Brazil",       pop=pop_br,  temps=temps_br),
        dict(name="South Africa", pop=pop_sa,  temps=temps_sa),
        dict(name="Antarctica",   pop=pop_ant, temps=temps_ant),
        dict(name="Death Valley", pop=pop_dv,  temps=temps_dv),
    ]

    # color gradient
    all_t = [s["temps"].to_numpy() for s in series]
    all_t_concat = np.concatenate(all_t)

    vmin = all_t_concat.min()
    vmax = all_t_concat.max()
    vcenter = all_t_concat.mean()  # or use 25 if you prefer a fixed ideal temp

    common_norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    common_cmap = "coolwarm"       # one colormap for all lines

    fig, ax = plt.subplots(figsize=(10, 6))

    all_y = []
    all_x_lengths = []

    # store line collections if you later want a single shared colorbar
    line_collections = []

    # plotting
    for idx, s in enumerate(series, start=1):
        # Extract data
        y = s["pop"]["total"].to_numpy()
        x = np.arange(len(y))
        t = s["temps"].to_numpy()

        all_y.append(y)
        all_x_lengths.append(len(y))

        # Build segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=common_cmap,
            norm=common_norm
        )
        lc.set_array(t[:-1])   # temperature per segment
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        line_collections.append(line)

    # ---------- AXES LIMITS ----------
    y_min = min(y.min() for y in all_y)
    y_max = max(y.max() for y in all_y)
    max_len = max(all_x_lengths)

    ax.set_xlim(0, max_len + 1)
    ax.set_ylim(y_min - 500, y_max + 500)

    # ---------- (OPTIONAL) SINGLE COLORBAR ----------
    # If you want one shared colorbar for all lines, uncomment this:
    cbar = plt.colorbar(line_collections[0], ax=ax)
    cbar.set_label("Temperature (°C or °F, depending on input)")

    # ---------- LEGEND WITH NUMBERS ----------
    legend_handles = []
    legend_labels = []
    for idx, s in enumerate(series, start=1):
        # simple black line in legend; numbering links to the description
        legend_handles.append(Line2D([0], [0], color="black", lw=3))
        legend_labels.append(f"{idx}: {s['name']}")

    ax.legend(legend_handles, legend_labels, title="Line numbering")

    # ---------- LABELS / TITLE ----------
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(
        "Bee Population Colored by Shared Temperature Gradient\n"
        "1: Germany, 2: Brazil, 3: South Africa, 4: Antarctica, 5: Death Valley (US)"
    )

    plt.show()

from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt

def plot_func_shared_gradient(pop_de, temps_de,
                              pop_br, temps_br,
                              pop_sa, temps_sa,
                              pop_ant, temps_ant,
                              pop_dv, temps_dv):
    """
    Plot bee population for 5 locations with a SINGLE shared temperature color gradient.
    All temperatures are mapped to the same colormap + normalization.
    """

    # Pack all series into a list to avoid repeating code
    series = [
        dict(name="Germany",      pop=pop_de,  temps=temps_de),
        dict(name="Brazil",       pop=pop_br,  temps=temps_br),
        dict(name="South Africa", pop=pop_sa,  temps=temps_sa),
        dict(name="Antarctica",   pop=pop_ant, temps=temps_ant),
        dict(name="Death Valley", pop=pop_dv,  temps=temps_dv),
    ]

    # ---------- GLOBAL COLOR SCALE (shared gradient for all lines) ----------
    # Concatenate all temperature arrays to compute a single vmin, vmax, vcenter
    all_t = [s["temps"].to_numpy() for s in series]
    all_t_concat = np.concatenate(all_t)

    vmin = all_t_concat.min()
    vmax = all_t_concat.max()
    vcenter = all_t_concat.mean()    # or use a fixed "ideal" temp if you prefer

    all_t = [s["temps"].to_numpy() for s in series]
    all_t_concat = np.concatenate(all_t)

    # Make sure the scale includes 25
    vmin = all_t_concat.min()
    vmax = all_t_concat.max()

    # Force the center to be 25°C
    vcenter = 25.0

    common_norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    common_cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(10, 6))

    all_y = []
    all_x_lengths = []
    line_collections = []


    # all lines in one axis set

    for s in series:
        # Extract data
        y = s["pop"]["total"].to_numpy()
        x = np.arange(len(y))
        t = s["temps"].to_numpy()

        all_y.append(y)
        all_x_lengths.append(len(y))

        # Build segments (so different parts of the line can have different colors)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(
            segments,
            cmap=common_cmap,
            norm=common_norm
        )
        lc.set_array(t[:-1])   # temperature per segment
        lc.set_linewidth(2)

        line = ax.add_collection(lc)
        line_collections.append(line)

    # ---------- AXES LIMITS ----------
    y_min = min(y.min() for y in all_y)
    y_max = max(y.max() for y in all_y)
    max_len = max(all_x_lengths)

    ax.set_xlim(0, max_len + 1)
    ax.set_ylim(y_min - 500, y_max + 500)

    # ---------- SINGLE COLORBAR FOR ALL LINES ----------
    cbar = plt.colorbar(line_collections[0], ax=ax)
    cbar.set_label("Temperature")

    # ---------- LEGEND (simple black lines, names only) ----------
    #legend_labels = []
    #legend_handles = []
    #   legend_handles.append(Line2D([0], [0], color="black", lw=3))
    #for s in series:
    #  legend_labels.append(s["name"])

    #ax.legend(legend_handles, legend_labels, title="Locations")

    country_name = s["name"]
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(str(country_name))

    plt.show()


def subplots(
    germany_data, germany_temp,
    brazil_data, brazil_temp,
    southAfrica_data, southAfrica_temp,
    antarctica_data, antarctica_temp,
    death_valley_data, death_valley_temp,
    common_cmap=plt.get_cmap("coolwarm"),
    names=("Germany", "Brazil", "South Africa", "Antarctica", "Death Valley (US)"),
):
    # Pack into a consistent iterable: (name, population_series, temperature_array_like)
    series = [
        (names[0], germany_data["total"], germany_temp),
        (names[1], brazil_data["total"], brazil_temp),
        (names[2], southAfrica_data["total"], southAfrica_temp),
        (names[3], antarctica_data["total"], antarctica_temp),
        (names[4], death_valley_data["total"], death_valley_temp),
    ]

    n = len(series)
    fig, axs = plt.subplots(nrows=n, ncols=1, figsize=(10, 2.5 * n), sharex=False)
    if n == 1:
        axs = [axs]

    for ax_i, (name, pop_total, temps) in zip(axs, series):
        # Convert to numpy
        y = np.asarray(pop_total)
        t = np.asarray(temps)
        x = np.arange(len(y))

        # Safety: line segments need at least 2 points
        if len(y) < 2 or len(t) < 2:
            ax_i.set_title(f"{name} (not enough data to plot)")
            ax_i.set_xlabel("Day")
            ax_i.set_ylabel("Population")
            continue

        # If temps length doesn't match, truncate to the shorter length
        m = min(len(y), len(t))
        y = y[:m]
        t = t[:m]
        x = x[:m]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Per-subplot scaling, always centered at 25°C
        tmin = float(np.nanmin(t))
        tmax = float(np.nanmax(t))
        eps = 1e-6
        vmin = min(tmin, 25.0 - eps)
        vmax = max(tmax, 25.0 + eps)
        norm_i = TwoSlopeNorm(vmin=vmin, vcenter=25.0, vmax=vmax)

        lc = LineCollection(segments, cmap=common_cmap, norm=norm_i)
        lc.set_array(t[:-1])     # temperature per segment
        lc.set_linewidth(2)

        ax_i.add_collection(lc)
        ax_i.set_xlim(0, len(x) - 1)
        ax_i.set_ylim(np.nanmin(y) - 500, np.nanmax(y) + 500)

        ax_i.set_ylabel("Population")
        ax_i.set_title(name)

        cbar = fig.colorbar(lc, ax=ax_i)
        cbar.set_label("Temperature (°C)")

    axs[-1].set_xlabel("Day")
    fig.suptitle("Each location (own temperature scaling, centered at 25°C)")
    fig.tight_layout()
    plt.show()


import numpy as np

def _align_series(pop_total, temps):
    """Align to same length and return 1D float arrays."""
    x = np.asarray(pop_total, dtype=float).ravel()
    y = np.asarray(temps, dtype=float).ravel()
    n = min(len(x), len(y))
    return x[:n], y[:n]

def distance_correlation(x, y):
    """
    Distance correlation in [0, 1]. 0 means independence (in ideal conditions),
    larger values mean stronger (possibly nonlinear) dependence.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    if n < 3:
        return np.nan

    # Pairwise distance matrices
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])

    # Double-center distance matrices
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).mean()
    dvarx2 = (A * A).mean()
    dvary2 = (B * B).mean()

    if dvarx2 <= 0 or dvary2 <= 0:
        return 0.0

    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx2 * dvary2)))

def max_lagged_pearson(x, y, max_lag=30):
    """
    Returns (best_corr, best_lag) where corr is Pearson corr between
    x[t] and y[t - lag]. lag=0 means same-day.
    Uses max absolute correlation across lags (common when sign may flip).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]

    best_corr = np.nan
    best_lag = 0

    for lag in range(0, min(max_lag, n - 2) + 1):
        # x aligned with past y
        x_seg = x[lag:]
        y_seg = y[:len(x_seg)]

        if len(x_seg) < 3:
            continue

        c = np.corrcoef(x_seg, y_seg)[0, 1]
        if np.isnan(c):
            continue

        if np.isnan(best_corr) or abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag

    return float(best_corr), int(best_lag)

def get_nonlin_metrics(pop_de, temps_de,
                       pop_br, temps_br,
                       pop_sa, temps_sa,
                       pop_ant, temps_ant,
                       pop_dv, temps_dv,
                       max_lag=30):

    print("getting nonlinear dependence (distance correlation) and lagged correlation")

    def compute(pop_df, temps):
        p, t = _align_series(pop_df["total"], temps)
        dcor = distance_correlation(p, t)
        best_corr, best_lag = max_lagged_pearson(p, t, max_lag=max_lag)
        return dcor, best_corr, best_lag

    de = compute(pop_de, temps_de)
    br = compute(pop_br, temps_br)
    sa = compute(pop_sa, temps_sa)
    ant = compute(pop_ant, temps_ant)  # auto-aligned like your Antarctica handling
    dv = compute(pop_dv, temps_dv)

    # Return in the same “one value per country” spirit, but now:
    # each country returns (dCor, best_lagged_corr, best_lag_days)
    return de, br, sa, ant, dv


def get_stats(temps_de, temps_br, temps_sa, temps_ant, temps_dv):
    """
    Get statistics for the temperatures of each location.
    Returns a dictionary with max, min, mean and std deviation for each location.
    """
    stats = {
        "Germany": {"mean": temps_de.mean(), "std": temps_de.std(), "max": temps_de.max(), "min": temps_de.min()},
        "Brazil": {"mean": temps_br.mean(), "std": temps_br.std(), "max": temps_br.max(), "min": temps_br.min()},
        "South Africa": {"mean": temps_sa.mean(), "std": temps_sa.std(), "max": temps_sa.max(), "min": temps_sa.min()},
        "Antarctica": {"mean": temps_ant.mean(), "std": temps_ant.std(), "max": temps_ant.max(), "min": temps_ant.min()},
        "Death Valley": {"mean": temps_dv.mean(), "std": temps_dv.std(), "max": temps_dv.max(), "min": temps_dv.min()},
    }
    return stats

# getting the data
germany_data, germany_temp = germany()
brazil_data, brazil_temp = brazil()
southAfrica_data, southAfrica_temp = south_africe()
antarctica_data, antarctica_temp = antarctica()
death_valley_data, death_valley_temp = death_valley()
print(brazil_data)
print(brazil_temp)
print(southAfrica_temp)

# getting  statistics including mean, std, max and min
stats = get_stats(germany_temp, brazil_temp, southAfrica_temp, antarctica_temp, death_valley_temp)
print("printing stats")
pprint(stats)

# Call the metric function
results = get_nonlin_metrics(
    germany_data, germany_temp,
    brazil_data, brazil_temp,
    southAfrica_data, southAfrica_temp,
    antarctica_data, antarctica_temp,
    death_valley_data, death_valley_temp,
    max_lag=30
)

countries = ["Germany", "Brazil", "South Africa", "Antarctica", "Death Valley"]

print("\nNonlinear temperature-population dependence metrics\n")
print(f"{'Country':<15} {'dCor':>8} {'Best lagged r':>14} {'Lag (days)':>12}")
print("-" * 55)

for country, (dcor, best_corr, best_lag) in zip(countries, results):
    print(f"{country:<15} {dcor:8.3f} {best_corr:14.3f} {best_lag:12d}")


# plot both on the same axes with different gradients
plot_func_shared_gradient(germany_data, germany_temp, brazil_data, brazil_temp, southAfrica_data, southAfrica_temp, antarctica_data, antarctica_temp, death_valley_data, death_valley_temp)
subplots(
    germany_data, germany_temp,
    brazil_data, brazil_temp,
    southAfrica_data, southAfrica_temp,
    antarctica_data, antarctica_temp,
    death_valley_data, death_valley_temp,)