from xmlrpc.client import Boolean
import  seaborn as sns
from    matplotlib import pyplot as plt
import  pandas as pd
import  numpy as np
import  math

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