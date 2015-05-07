function [ns_in_cl, clusts, clusters]= get_nc_clusters_from_SL_dendro(Z,nc,ns)
% retorna os clusters correspondentes a uma particao com nc clusters

[H,clusters] = dendrogram(Z,nc);
close; %para apagar o plot do dendrograma
clusts=[];
for k = 1 : nc
   a=find(clusters==k);
   ns_in_cl(k)=length(a);
   clusts(k,1:ns_in_cl(k))=a(1:ns_in_cl(k))';
end
