from re import X
import seaborn as sns
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import random
import matplotlib as mpl
from   matplotlib import pyplot as plt
from   pathlib import Path

from xmlrpc.client import Boolean
from typing import List, Tuple, Dict, Optional


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.family'] = 'Arial'
figsize=(10,6)


def countplot(
    data: pd.DataFrame, 
    x: List[str],
    subplots: Boolean = True, 
    ncols: int = 2,    
    maxitems = 10, 
    other_category = True,
    title: str = '',  
    name: str = 'countplot',
    figurepath: Path = Path("../reports/figures")
):

    if subplots:
        number_of_items = len(x)
        nrows = math.ceil(number_of_items / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        
        if title:
            fig.suptitle(title)

        for item, ax in zip (x, axes.flatten()):

            ax.title.set_text(item)
            s = data[item]             
            if maxitems:                   
                d = s.value_counts()[:maxitems]
                sns.barplot(x=d.values, y=d.index, ax = ax, orient = 'h')                    
            else:                
                d = s.value_counts()               
                sns.barplot(x=d.values, y=d.index, ax = ax, orient = 'h')
                            
            ax.set(xlabel="", ylabel = "")
    else:
        ax = sns.countplot(data = data, y = x[0], order = data[x[0]].value_counts().index)
        ax.set(xlabel="", ylabel = "")
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  


def boxplot(
    data: np.ndarray,
    x: str,
    y: str,    
    hue: str = '',
    figsize: Tuple[int, int] = figsize,
    name: str = 'boxplot',
    figurepath: Path = Path("../reports/figures")    
) -> None:
    plt.figure(figsize=figsize)
    if hue:
        sns.boxplot(data=data, x=x, y=y, hue=hue)
    else:
        sns.boxplot(data=data, x=x, y=y)
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  
    
    # plt.xticks(rotation=90)


def lineplot(
    data: pd.DataFrame,    
    x: str,
    y: str,    
    hue: str = '',
    figsize: Tuple[int, int] = figsize, 
    name: str = 'lineplot',
    figurepath: Path = Path("../reports/figures")   
) -> plt.Axes:  

    plt.figure(figsize=figsize)    
    ax = sns.lineplot(data = data, x = x, y = y, hue =  hue)
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  

    return ax

def timeplot(
    train: pd.Series,
    test: pd.Series,
    figsize: Tuple[int, int] = figsize,   
    name: str = 'timeplot',
    figurepath: Path = Path("../reports/figures") 
) -> plt.Axes:  
    plt.figure(figsize=figsize) 
    ax = sns.lineplot(data = train, x = train.index, y = train.values)
    sns.lineplot(data = test, x = test.index, y = test.values, ax = ax)
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  
    return ax
    


def scatterplot(
    data: pd.DataFrame,    
    x: str,
    y: str,    
    hue: str = '',
    label: str = '',
    figsize: Tuple[int, int] = figsize,
    name: str = 'scatterplot',
    figurepath: Path = Path("../reports/figures")
) -> plt.Axes:  
    
    plt.figure(figsize=figsize)    
    ax = sns.scatterplot(data = data, x = x, y = y, hue = hue, label = label)
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  

    return ax

def boxplot(
    data: pd.DataFrame,    
    x: str,
    y: str,            
    order: List[str] = [],
    figsize: Tuple[int, int] = figsize,
    name: str = 'boxplot',
    figurepath: Path = Path("../reports/figures")
) -> plt.Axes:  
    
    plt.figure(figsize=figsize)        
    ax = sns.boxplot(data = data, x = x, y = y, order = order)    
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  

    return ax

def distribution(
    data: pd.DataFrame,
    x: str,    
    xlabel: str = 'Prijs',
    ylabel: str = 'Aantal',
    figsize: Tuple[int, int] = figsize,
    name: str = 'distribution',
    figurepath: Path = Path("../reports/figures")
) -> plt.Axes:  

    plt.figure(figsize=figsize) 
    ax = sns.histplot(data, x = x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  

    return ax

def plot_example(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    examples: int,
    model: Optional[tf.keras.Model] = [],
    figsize: Tuple[int, int] = (10, 3),
    name: str = 'model_example',
    figurepath: Path = Path("../reports/figures")
) -> plt.Figure:
    ''' plot examples from timeseries model with predictions '''
    
    x, y = next(iter(data))      
    yhat = []  
    if model:
        yhat = model.predict(data)   

    # adjust figure size depending on number of examples
    w,h = figsize
    figsize = (w, h * examples)
    
    # Calculate x axis to show window + horizon
    input_slice = slice(0, window_size)
    total_window_size = window_size + horizon
    input_indices = np.arange(total_window_size)[input_slice]
    label_start = total_window_size - horizon
    labels_slice = slice(label_start, None)
    label_indices = np.arange(total_window_size)[labels_slice]

    # Choice x samples without replacement
    samples = np.random.choice(np.arange(0,len(x)),examples, replace=False)

    fig, axes = plt.subplots(nrows=examples, ncols= 1,sharey=True, sharex=True, figsize=figsize)
    fig.tight_layout()
    for sample, ax in zip(samples, axes.flatten()):
    
        if x.ndim > 2: # More features : Eerste feature is het signaal
            ax.plot(input_indices, x[sample][:,0], marker='.', c='blue')
        else:
            ax.plot(input_indices, x[sample], marker='.', c='blue')

        ax.plot(label_indices, y[sample], marker='.', c='green')
        ax.scatter(label_indices, y[sample], edgecolors='k', label='Labels', c='green', s=64)
        if np.any(yhat):
            ax.scatter(label_indices, yhat[sample], marker='X', edgecolors='k', label='predictions', c='#ff7f0e', s=64)

    plt.savefig(Path(figurepath, name + '.svg'),  bbox_inches = 'tight')  

    return fig









