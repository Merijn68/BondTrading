import seaborn as sns
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import random
from   matplotlib import pyplot as plt
from xmlrpc.client import Boolean
from typing import List, Tuple, Dict, Optional

def countplot(
    data: pd.DataFrame, 
    x: List[str],
    subplots: Boolean = True, 
    ncols: int = 2,    
    maxitems = 10, 
    other_category = True,
    title: str = '',  
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

def grouped_lineplot(
    data: pd.DataFrame,    
    ncols: int,
    x: str,    
    y: str,
    hue: str,
    group: str, 
    title: str = '',  
    figsize: Tuple[int, int] = (8, 6),  
) -> plt.Figure:

    grouplist = data[group].unique()
    nrows = math.ceil(len(grouplist) / ncols)
    print (ncols, nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,sharey=True, sharex=True, figsize=figsize)
    
    if title:
        fig.suptitle(title)
    
    for item, ax, in zip(grouplist, axes.flatten()):                          
            ax.title.set_text(item)
            selection = data[group] == item
            sns.lineplot(ax=ax, data=data[selection], x=x, y=y, hue = hue, legend = None)
            
    return fig


def boxplot(
    data: np.ndarray,
    x: str,
    y: str,    
    hue: str = '',
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    plt.figure(figsize=figsize)
    if hue:
        sns.boxplot(data=data, x=x, y=y, hue=hue)
    else:
        sns.boxplot(data=data, x=x, y=y)
    
    # plt.xticks(rotation=90)

def lplot(
    data: List[pd.DataFrame],        
    figsize: Tuple[int, int] = (8, 6),
) -> None:    
    palletes = ['green', 'blue', 'red']
    plt.figure(figsize=figsize)    
    x = 0
    for line in data:           
 
        sns.lineplot(data = line, palette = [palletes[x]])
        x = x + 1

def lineplot(
    data: pd.DataFrame,    
    x: str,
    y: str,    
    hue: str = '',
    figsize: Tuple[int, int] = (15, 8),    
) -> plt.Axes:  

    plt.figure(figsize=figsize)    
    ax = sns.lineplot(data = data, x = x, y = y, hue =  hue)

    return ax


def scatterplot(
    data: pd.DataFrame,    
    x: str,
    y: str,    
    hue: str = '',
    label: str = '',
    figsize: Tuple[int, int] = (15, 8),
) -> plt.Axes:  
    
    plt.figure(figsize=figsize)    
    ax = sns.scatterplot(data = data, x = x, y = y, hue = hue, label = label)

    return ax

def boxplot(
    data: pd.DataFrame,    
    x: str,
    y: str,            
    order: List[str] = [],
    figsize: Tuple[int, int] = (15, 8),
) -> plt.Axes:  
    
    plt.figure(figsize=figsize)        
    ax = sns.boxplot(data = data, x = x, y = y, order = order)    
    
    return ax

def plot_example(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    examples: int,
    model: Optional[tf.keras.Model] = [],
    figsize: Tuple[int, int] = (11, 3),
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
    
        ax.plot(input_indices, x[sample], marker='.', c='blue')
        ax.plot(label_indices, y[sample], marker='.', c='green')
        ax.scatter(label_indices, y[sample], edgecolors='k', label='Labels', c='green', s=64)
        if np.any(yhat):
            ax.scatter(label_indices, yhat[sample], marker='X', edgecolors='k', label='predictions', c='#ff7f0e', s=64)

    return fig









