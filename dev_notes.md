


# Pandas

```
for key, grp in df.groupby(level=['index1']):
    plt.plot(grp['col1'],label=key)
```
`grp.reset_index(level=0,inplace=True)` puts indeces as columns.
`grp.index.levels[i]` gets a list of the values of the index in level _i_




Does it make sense to use lifetimes with MST SL? As it is the lifetimes approach with MST has a problem. Even the normal one might have a problem, not sire. When checking for the maximum lifetime, I'm pickig the first one that shows up. In the case that there are several maxima I still pick the first one, which might mean a LOT of edge cutting. E.g. in the 10000 gaussian dataset example I get the first maximum around edge 1200 and that would mean I would cut around 8400 edges. Which would then produce ~8400 clusters. This is not wanted. On the other hand, if I pick the last one (the closer to the maximum weights) I would only get 4 extra clusters. This is a big difference and the latter probably yields much better results.


I'm gonna do a study about the relationship between the number of samples, k interval