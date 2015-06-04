ficheiro{1}='rings.txt';
inicio_{1}=[1,201,401];
fim_{1}=[200,400,450];
nc_{1}=3;
kmin_{1}=10;
kmax_{1}=30;
iter_{1}=150;

l=1;

fich=ficheiro{l};
n_c=nc_{l};
inicio=inicio_{l};
fim=fim_{l};
kmin = kmin_{l};
kmax = kmax_{l};
iter= iter_{l};

dados=LeMatriz_s(fich);
[num_amostras,features]=size(dados);

%Ground truth
v_trueclass=fim-inicio+1;
trueclass=zeros(length(v_trueclass),max(v_trueclass));
for i=1:length(v_trueclass)
    trueclass(i,1:v_trueclass(i))=[inicio(i):fim(i)];
end

%matriz de distancia
Y=pdist(dados);
R=squareform(Y);

%teste com MST da BGL
Rsparse=sparse(R);
T = mst(Rsparse);
figure;hold on;plot(dados(:,1),dados(:,2),'.r');gplot(T,dados,'b-');grid
title('Minimum Spanning Three (MST) BGL (diss)')

%teste com MST codigo Matlab
figure;
[adj_l, adj_l_cost, Xmst,D] = mste_new(R,dados);
%title('Minimum Spanning Three (MST) mste_new (diss)')

%matriz de semelhanca
S=max(max(R))-R;
%teste com MST codigo Matlab (matriz de semelhanca)
figure;
%[adj_l, adj_l_cost, Xmst,XmstValue] = mste_new_S(S,dados);
[T,cost] = mste_new_s2(S,dados);


[T, cost,MSTreeEdges] =MST_Kruskal2(S);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall (before cut)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,T);

[Sc,clustersMST]= extract_K_mste2(T,n_c,num_amostras);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall (after cut-extract_K_mste2)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);

[Sc,clustersMST]= extract_K_mste3(T,n_c,num_amostras,MSTreeEdges);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall (after cut-extract_K_mste3)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);

%% criterio life-time
figure;plot(MSTreeEdges(:,1));
pesos=flipud(MSTreeEdges(:,1))
dif=diff(pesos);
figure;plot(dif)
[maximodif, maxdifI]=max(dif);


[Sc,clustersMST]= extract_K_mste3_lifeTime(T,num_amostras,MSTreeEdges);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall (after cut-extract_K_mste3_lifeTime)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);


%% cria ensemble 
ensemble=[];
for iiiter=1:iter
    k=round(unifrnd(kmin,kmax));
    %inicializa aleatoriamente os centroides
    centroids_seed=random_k_seed(k,num_amostras);
    [nsamples_in_cluster,clusters_m]=k_medias_with_seed_vns(dados,k,centroids_seed);
    ensemble(iiiter).nsamples_in_cluster=nsamples_in_cluster;
    ensemble(iiiter).clusters_m=clusters_m;
    fprintf(1,'.')
end
fprintf(1,'\n')
%% cria co-assocs e realiza extracao usando EAC
methods={'single','complete','average','ward','centroid'};
%extracao classica
[out]=combina_generico2_sparse(ensemble,num_amostras,[],methods,n_c,v_trueclass,trueclass);
%agora usando extracao baseada na MST
[outMST,clusters,assocs]=combina_generico2_sparseMST2(ensemble,num_amostras,[],methods,n_c,v_trueclass,trueclass);

figure;imagesc(assocs);figure;spy(assocs);

figure;hold on;plot(dados(:,1),dados(:,2),'.r');gplot(outMST,dados,'b-');grid;title('MST ja cortada baseada nas co-assocs (BGL)')

[i,j,s]=find(assocs);
[IsMax]=find(s==1);
s=1-s;
delta=eps;s(IsMax)=delta;
assocsout=sparse(i,j,s,num_amostras,num_amostras);
T = mst(assocsout);
figure;hold on;plot(dados(:,1),dados(:,2),'.r');gplot(T,dados,'b-');grid;title('MST baseada nas co-assocs (BGL)')


%teste com MST codigo Matlab
figure;
[adj_l, adj_l_cost, Xmst,D] = mste_new(assocsout,dados);

%teste com MST codigo Matlab (matriz de semelhanca)
figure;
[adj_l, adj_l_cost, Xmst,XmstValue] = mste_new_s(assocs,dados);
figure;
[adj_l, adj_l_cost, Xmst,XmstValue] = mste_new_s(assocs,dados,num2str([1:num_amostras]'));

%cria matriz de adjacencia esparsa a custa de vectores c indices e pesos
T=sparse(Xmst(:,1),Xmst(:,2),XmstValue,num_amostras,num_amostras);
T=T+T'; %torna a matriz simetrica

%usa adaptacao do extract_K_MST2 para semelhancas
[S,clusters,verticesCortados]=extract_K_mste(T,3,num_amostras); 
figure;hold on;plot(dados(:,1),dados(:,2),'.r');gplot(S,dados,'b-');
grid;title('Minimum Spanning Three (MST) for data table (based on similarities) - after cut')

%% agora usando Kruskal

[Tc, cost,MSTreeEdges] =  MST_Kruskal2(assocs);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall - Assocs (before cut)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Tc);

[Ss,clustersMST]= extract_K_mste2(Tc,n_c,num_amostras);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall - Assocs  (after cut)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);


[Sc,clustersMST]= extract_K_mste3(Tc,n_c,num_amostras,MSTreeEdges);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall - Assocs(after cut-extract_K_mste3)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);

[Sc,clustersMST]= extract_K_mste3_lifeTime(Tc,num_amostras,MSTreeEdges);
figureh=figure;holds=0;
titulo=['Minimum Spanning Three (MST) based on Kruskall - Assocs(after cut-extract_K_mste3_lifeTime)'];
plot_clustering(dados,clustersMST,figureh,titulo,holds,Sc);
