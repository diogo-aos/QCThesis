11-04-2015

Running the some quick tests in the *mariana* computer at INESC-ID I realized that the speedup was quite bad (<2). This, using the algorithm that computes the entire distance matrix. One of the big underlying problems is having an assignment time very big. The implementation of this part involved copying the whole dataset and rearranging it. Even taking only in consideration the optimized component, the speedup was low (<10).

I decided to change the way to compute things. Instead of computing the whole distance matrix and label the data points afterwards, the algorithm now labels mediately. The kernel for each data point computes the best label. This is the way it's done in some articles (it was a big mistake not reading more thoroughly the literature) and much more efficient. The speedup in the optimized part of the code is huge comparing with NumPy itself (>400). There is a HUGE bottleneck in the centroid recomputation. Even though the first part is much faster, overall it's slower that the previous implementation.

12-04-2015
How is it possible that there are centroids that are not assigned 
in the first iteration? The initial centroids are picked from the 
data. That means that each centroid will have a distance of 0 to at 
least one datum, and that datum should be assigned for sure to that 
centroid!

# Pandas

```
for key, grp in df.groupby(level=['index1']):
    plt.plot(grp['col1'],label=key)
```
`grp.reset_index(level=0,inplace=True)` puts indeces as columns.
`grp.index.levels[i]` gets a list of the values of the index in level _i_
