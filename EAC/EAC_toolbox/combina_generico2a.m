function combina_generico2a(files_prefix,ns,vector_fich,methods,n_c,directo)
%Sintax:
%combina_generico2a(files_prefix,ns,vector_fich,methods,n_c,directo)
%inputs--------------------------------------------------------------------
%files_prefix:  Target file prefix.
%ns:			number of samples
%vector_fich: 	List of file names with the partitions to combine. For n
%               partitions vector_fich is a cell array with n objects.
%methods: 		List of hierarchical methods used for the extraction of the final partition
%               Possible methods:
%					'single'   --- nearest distance 
%       			'complete' --- furthest distance
%       			'average'  --- average distance
%       			'centroid' --- center of mass distance
%       			'ward'     --- inner squared distance
%					if method=[] then SL is used
%n_c:			final number of clusters
%directo        Target directory
%outputs-------------------------------------------------------------------
%The combination results are stored in .mat files.Two files produced for each method:
%Life time criteria: [files_prefix method '-Stable-combined.mat']
%Fixed k: [files_prefix method '-fixed-k-Stable-combined.mat']
%Also the co-association matrix is saved in a .mat file. ([files_prefix 'nassoc.mat'])
%--------------------------------------------------------------------------
% Written by:
%   Ana Fred & Andre Lourenco
%   Instituto Superior Tecnico
%   1049-001 Lisboa
%   Portugal
% 

%n_clusterings - n� de ficheiros
n_clusterings=length(vector_fich);

assocs=zeros(ns,ns);

for iter=1:n_clusterings
   nsamples_in_cluster=[];
   clusters_m=[];
   fichaux = [vector_fich{iter}]
   load(fichaux,'nsamples_in_cluster','clusters_m')
   
   assocs=update_assoc_mats3b(assocs,nsamples_in_cluster,clusters_m);
end

%-------------------- normalizacao ---------------------------
nassocs=assocs/n_clusterings;
%-------------------------------------------------------------

if exist([directo '/eac'],'dir')
    cd([directo '/eac'])
else
    mkdir(directo,['eac'])
    cd([directo '/eac'])
end

%-------------------- visualizacao assocs --------------------
imagesc(nassocs)
colormap(jet)
colorbar
print(gcf,'-depsc','-noui',[files_prefix 'assocs' '.eps']);
print(gcf,'-djpeg','-noui',[files_prefix 'assocs' '.jpg']);

%-------------------------------------------------------------

save([files_prefix 'nassoc.mat'],'nassocs');

%-------------------- extraccao da particao final ------------
%para usar com qualquer m�todo hierarquico para obten��o do cluster final.
for o=1:length(methods)
    method=methods{o};
    cd([directo])
    Z=apply_hierq2nassocs1(nassocs,method);
    cd([directo '/eac'])
    files_prefix2=[files_prefix method];
    %k-livre
    [nc_stable, nsamples_in_cluster, clusters_m ]= get_nc_stable_from_SL_dendro(Z, ns);
    save([files_prefix2 '-Stable-combined.mat'],'nsamples_in_cluster','clusters_m');
    nsamples_in_cluster=[];
    clusters_m=[];
    n_c_livre=length(nsamples_in_cluster);
    %k-fixo
    [nsamples_in_cluster,clusters_m]= get_nc_clusters_from_SL_dendro(Z,n_c,[]);
    save([files_prefix2 '-k-fixo-Stable-combined.mat'],'nsamples_in_cluster', 'clusters_m');
end
%-------------------------------------------------------------



