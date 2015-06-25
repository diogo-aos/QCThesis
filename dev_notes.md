


# Pandas

```
for key, grp in df.groupby(level=['index1']):
    plt.plot(grp['col1'],label=key)
```
`grp.reset_index(level=0,inplace=True)` puts indeces as columns.
`grp.index.levels[i]` gets a list of the values of the index in level _i_




For some reason GPU Boruvka doesn't work when I give device arrays as input. The weird thing is the first error is transferring the convergence variable in the propagate colors cycle. I'm gonna have to check the other arrays before getting to that point and see if there is something weird. But, in the end, it doesn't really matter because GPU Boruvka is so much slower than the sequential counterpart with this graph.