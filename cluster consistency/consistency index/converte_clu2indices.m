function indices=converte_clu2indices(nsamples_in_cluster,clusters_m)

numero_clusters=length(nsamples_in_cluster);
numero_max_amostras=max(nsamples_in_cluster);

indices=zeros(numero_clusters,numero_max_amostras);

for i=1:numero_clusters
    if nsamples_in_cluster(i) ~= 0
        indices(i,1:nsamples_in_cluster(i))=find(clusters_m==i)';
    end
end