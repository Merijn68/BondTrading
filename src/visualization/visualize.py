"""
    Time series visualizations
"""

import seaborn as sns
import pandas as pd
import numpy as np
import math
import tensorflow as tf

import re
import matplotlib as mpl
from matplotlib import pyplot as plt
from pathlib import Path

from xmlrpc.client import Boolean
from typing import List, Tuple, Optional, Dict


# Generic setup parameters for Matplotlib
figsize = (10, 6)
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["figure.figsize"] = figsize


def countplot(
    data: pd.DataFrame,
    x: List[str],
    subplots: Boolean = True,
    ncols: int = 2,
    maxitems=10,
    title: str = "",
    name: str = "countplot",
    figsize: Tuple[int, int] = figsize,
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """Grouped count (bar) plot to show categorical data"""

    if subplots:
        number_of_items = len(x)
        nrows = math.ceil(number_of_items / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        if title:
            fig.suptitle(title)

        for item, ax in zip(x, axes.flatten()):

            ax.title.set_text(item)
            s = data[item]
            if maxitems:
                d = s.value_counts()[:maxitems]
                sns.barplot(x=d.values, y=d.index, ax=ax, orient="h")
            else:
                d = s.value_counts()
                sns.barplot(x=d.values, y=d.index, ax=ax, orient="h")

            ax.set(xlabel="", ylabel="")
    else:
        ax = sns.countplot(data=data, y=x[0], order=data[x[0]].value_counts().index)
        ax.set(xlabel="", ylabel="")
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def boxplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "",
    order: List[str] = [],
    figsize: Tuple[int, int] = figsize,
    name: str = "boxplot",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """simple box plot"""

    plt.figure(figsize=figsize)
    if not order:
        order = data.groupby(by=[x])[y].median().iloc[::1].index
    if hue:
        sns.boxplot(data=data, x=x, y=y, hue=hue)
    else:
        ax = sns.boxplot(data=data, x=x, y=y, order=order)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")

    return ax


def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_label: str = "",
    y_label: str = "",
    hue: str = "",
    hue_order: List = [],
    figsize: Tuple[int, int] = figsize,
    name: str = "lineplot",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """simple line plot"""

    plt.figure(figsize=figsize)
    if hue_order:
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order)
    else:
        ax = sns.lineplot(data=data, x=x, y=y, hue=hue)

    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)

    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def lineplot2(
    data: pd.DataFrame,
    data2: pd.DataFrame,
    x: str,
    y: str,
    y2: str,
    x_label: str,
    y_label: str,
    y2_label: str,
    label: str,
    label2: str,
    figsize: Tuple[int, int] = figsize,
    invert_yaxis2: Boolean = False,
    name: str = "doublelineplot",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """double y axis line plot"""

    plt.figure(figsize=figsize)

    ax = sns.lineplot(data=data, x=x, y=y, label=label, color="black")
    ax2 = plt.twinx()
    sns.lineplot(data=data2, x=x, y=y2, label=label2, ax=ax2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.get_legend().remove()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax2.set_ylabel(y2_label)

    if invert_yaxis2:
        ax2.invert_yaxis()

    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")

    return ax


def timeplot(
    train: pd.Series,
    test: pd.Series,
    x_label: str = "",
    y_label: str = "",
    figsize: Tuple[int, int] = figsize,
    name: str = "timeplot",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """Plot train and Test data in one plot"""

    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=train, x=train.index, y=train.values)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    sns.lineplot(data=test, x=test.index, y=test.values, ax=ax)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")
    return ax


def scatterplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "",
    label: str = "",
    figsize: Tuple[int, int] = figsize,
    name: str = "scatterplot",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """Simple scatterplot"""

    plt.figure(figsize=figsize)
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue, label=label)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")

    return ax


def distribution(
    data: pd.DataFrame,
    x: str,
    xlabel: str = "Prijs",
    ylabel: str = "Aantal",
    figsize: Tuple[int, int] = figsize,
    name: str = "distribution",
    figurepath: Path = Path("../reports/figures"),
) -> plt.Axes:
    """Plot a distribution (histogram)"""

    plt.figure(figsize=figsize)
    ax = sns.histplot(data, x=x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(Path(figurepath, name + ".svg"), bbox_inches="tight")

    return ax


def plot_example(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    examples: int,
    model: Optional[tf.keras.Model] = [],
    figsize: Tuple[int, int] = (10, 3),
    figurepath: Path = Path("../reports/figures"),
) -> plt.Figure:
    """plot examples from timeseries model with predictions"""

    x, y = next(iter(data))
    yhat = []
    if model:
        yhat = model.predict(data)

    # adjust figure size depending on number of examples
    w, h = figsize
    figsize = (w, h * examples)

    # Calculate x axis to show window + horizon
    input_slice = slice(0, window_size)
    total_window_size = window_size + horizon
    input_indices = np.arange(total_window_size)[input_slice]
    label_start = total_window_size - horizon
    labels_slice = slice(label_start, None)
    label_indices = np.arange(total_window_size)[labels_slice]

    # Choice x samples without replacement
    samples = np.random.choice(np.arange(0, len(x)), examples, replace=False)

    fig, axes = plt.subplots(
        nrows=examples, ncols=1, sharey=True, sharex=True, figsize=figsize
    )
    fig.tight_layout()
    fig.suptitle(model.name)
    for sample, ax in zip(samples, axes.flatten()):

        if x.ndim > 2:  # More features : Eerste feature is het signaal
            series = x[sample][:, 0]
        else:
            series = x[sample]
        ax.plot(input_indices, series, marker=".", c="blue", label="x waarde")
        ax.plot(label_indices, y[sample], marker=".", c="green", label="y waarde")
        ax.scatter(label_indices, y[sample], edgecolors="k", c="green", s=64)
        if np.any(yhat):
            ax.scatter(
                label_indices,
                yhat[sample],
                label="Voorspelling",
                marker="X",
                edgecolors="k",
                c="#ff7f0e",
                s=64,
            )

        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, loc="upper left")

    fig.tight_layout()
    plt.savefig(Path(figurepath, model.name + ".svg"), bbox_inches="tight")

    return fig


def plot_results(
    result: Dict,
    ymin: float = 0.0,
    ymax: Optional[float] = None,
    yscale: str = "linear",
    moving: int = -1,
    alpha: float = 0.5,
    patience: int = 1,
    subset: str = ".",
    grid: bool = False,
    measure: str = "loss",
    figsize: Tuple[int, int] = (15, 10),
) -> None:

    val_measure = "val_" + measure
    if not grid:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        if moving < 0:
            move = True

        for key in result.keys():
            if bool(re.search(subset, key)):
                ms: List[float] = result[key].history[measure]
                if move:
                    z = movingaverage(ms, moving)
                    z = np.concatenate([[np.nan] * moving, z[moving:-moving]])
                    color = next(ax1._get_lines.prop_cycler)["color"]
                    ax1.plot(z, label=key, color=color)
                    ax1.plot(ms, label=key, alpha=alpha, color=color)
                else:
                    ax1.plot(ms, label=key)

                ax1.set_yscale(yscale)
                ax1.set_ylim(ymin, ymax)
                ax1.set_title("train")
                valms = result[key].history[val_measure]

                if move:
                    z = movingaverage(valms, moving)
                    z = np.concatenate([[np.nan] * moving, z[moving:-moving]])[
                        :-patience
                    ]
                    color = next(ax2._get_lines.prop_cycler)["color"]
                    ax2.plot(z, label=key, color=color)
                    ax2.plot(valms, label=key, alpha=alpha, color=color)
                else:
                    ax2.plot(valms[:-patience], label=key)

                ax2.set_yscale(yscale)
                ax2.set_ylim(ymin, ymax)
                ax2.set_title("valid")

        plt.legend()
    if grid:

        keyset = list(filter(lambda x: re.search(subset, x), [*result.keys()]))
        gridsize = int(np.ceil(np.sqrt(len(keyset))))

        plt.figure(figsize=(15, 15))
        for i, key in enumerate(keyset):
            plt.subplot(gridsize, gridsize, i + 1)
            ms = result[key].history[measure]
            valms = result[key].history[val_measure]
            plt.plot(ms, label="train")
            plt.ylim(0, ymax)
            plt.plot(valms, label="valid")
            plt.title(key)
            plt.legend()


def movingaverage(interval: List[float], window_size: int = 32) -> List[float]:
    ret = np.cumsum(interval, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1 :] / window_size


def naive(result: Dict, ylim: float = 2, subset: str = ".") -> None:
    """Simple plot to show results from model"""

    for key in result.keys():
        if bool(re.search(subset, key)):
            plt.plot(result[key].history["val_smae"], label=key)

    plt.axhline(1.0, color="r", label="naive norm")
    plt.ylim(0, ylim)
    plt.legend()
