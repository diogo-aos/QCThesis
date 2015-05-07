%this is demo.m


%file that contains the data. Format:......
fich='cigar_data.txt';
%directory where the data is located.
directo='/home/chiroptera/workspace/QCThesis/EAC/EAC_toolbox/';  
cd(directo)
%number of natural clusters 
n_c=4;

%k-means, upper and lower bound for the number of clusters
kmin = 10;
kmax = 30;
%number of clusterings
iter= 30;

%reads matrix
dados=LeMatriz_s(fich);
[num_amostras,features]=size(dados);

%--------------------------------------------------------------------
%generation of the partitions

distribution = 'unif';
for iiiter=1:iter
    %generates one value for k (number of clusters)
    if distribution == 'gaus'
        k=round(normrnd(mu,sigma));
        if k < kmin
            k = kmin;
        else
            if k > kmax
                k = kmax;
            end
        end
    else
        k=round(unifrnd(kmin,kmax));
    end
    
    %k-means random initialization 
    centroids_seed=random_k_seed(k,num_amostras);
    %k-means - just one iteration
    [nsamples_in_cluster,clusters_m]=k_medias_with_seed_vns(dados,k,centroids_seed);
    save(['kmeans-' fich(1:end-4) '-' num2str(iiiter) '.mat'],...
        'nsamples_in_cluster','clusters_m');
end
%--------------------------------------------------------------------

%------------------------EAC-----------------------------------------
vector_fich=[];
for i=1:iter
    vector_fich{i}=['kmeans-' fich(1:end-4) '-' num2str(i) '.mat'];
end

files_prefix=[fich(1:end-4) '-eac-kmeans-'];
methods{1}='single'
methods{2}='complete'
methods{3}='average'
methods{4}='ward'
methods{5}='centroid'

combina_generico2a(files_prefix,num_amostras,vector_fich,methods,n_c,directo);
%--------------------------------------------------------------------