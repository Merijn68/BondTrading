import  seaborn as sns
from    matplotlib import pyplot as plt
import  pandas as pd
import  numpy as np
import  math

from typing import List, Tuple, Dict, Optional

def countplot(
    data: pd.DataFrame, 
    x: str,
    hue: str = ''  
):
    if hue:
        ax = sns.countplot(data = data, y = x, hue = hue, order = data[x].value_counts().index)
    else:
        ax = sns.countplot(data = data, y = x, order = data[x].value_counts().index)
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
    plt.figure(figsize=figsize)    
    for line in data:           
        sns.lineplot(data = line)
        
