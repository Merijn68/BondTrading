# def grouped_lineplot(
#     data: pd.DataFrame,    
#     ncols: int,
#     x: str,    
#     y: str,
#     hue: str,
#     group: str, 
#     title: str = '',  
#     figsize: Tuple[int, int] = (8, 6),  
# ) -> plt.Figure:

#     grouplist = data[group].unique()
#     nrows = math.ceil(len(grouplist) / ncols)
#     print (ncols, nrows)

#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols,sharey=True, sharex=True, figsize=figsize)
    
#     if title:
#         fig.suptitle(title)
    
#     for item, ax, in zip(grouplist, axes.flatten()):                          
#             ax.title.set_text(item)
#             selection = data[group] == item
#             sns.lineplot(ax=ax, data=data[selection], x=x, y=y, hue = hue, legend = None)
            
#     return fig


# def lplot(
#     data: List[pd.DataFrame],        
#     figsize: Tuple[int, int] = (8, 6),
# ) -> None:    
#     palletes = ['green', 'blue', 'red']
#     plt.figure(figsize=figsize)    
#     x = 0
#     for line in data:           
 
#         sns.lineplot(data = line, palette = [palletes[x]])
#         x = x + 1