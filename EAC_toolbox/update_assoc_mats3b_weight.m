function assocs=update_assoc_mats3b_weight(assocs,nsamples_in_clusters,clusters,weight);
%updates the co-assotiation matrix with the last clustering
%inputs--------------------------------------------------------------------
%assocs:                co-assotiation matrix
%nsamples_in_clusters:  line vector with the number of samples in each cluster
%clusters:              one cluster per line. Each line has the indices of the samples in each cluster. 
%                       (It is not necessary that these indeces are sorted)                       
%--------------------------------------------------------------------------


[nclusters,cols]=size(clusters);

for i=1:nclusters
    nsi=nsamples_in_clusters(i); 
    if nsi>1  %cluster com mais do que uma amostra
        assocs(clusters(i,1:nsamples_in_clusters(i)),clusters(i,1:nsamples_in_clusters(i)))=...
            assocs(clusters(i,1:nsamples_in_clusters(i)),clusters(i,1:nsamples_in_clusters(i)))+ ...
            weight.*ones(nsamples_in_clusters(i),nsamples_in_clusters(i));
    end
end    
