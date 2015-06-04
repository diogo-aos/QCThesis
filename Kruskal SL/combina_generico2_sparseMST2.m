function [T,clusters,assocs,verticesCortados]=combina_generico2_sparseMST2(ensemble,ns,neighbors,methods,n_c,v_trueclass,trueclass)
%% combina_generico2_sparseMST2
% algoritmo de combinacao de clusters EAC usando matrizes esparsas e
% extraccao usando MST 
% _Andre (versao 1: 28 Marco 2009)_
% _Andre (versao 1a: 22 Abril 2009)_ correccao: [out,representante]=extract_K_MST(T,n_c);
% _Andre (versao 2a: 30 Abril 2009)_ actualizacao da criacao da matriz esparsa co-assocs
% _Andre (versao 2a: 30 Abril 2009)_ actualizacao da extraccao da particao final extract_K_MST2(T,n_c);
% _Andre (versao 2b: 15 Set 2009)_ actualizacao matriz de coassocs (incluindo threshold no arcos do grafo com coasssocs = 1)
%%  inputs
%  * ensemble: 	
%   estrutura do tipo struct que contem dois campos: clusters_m_ e nsamples_in_cluster_ cada um destes campos e uma cell para poder albergar vectores de tamanho diference (em vez de ler a lista de nomes vector_fich)
%   ns: numero de instancias 
%   * neighbors:		matriz nsXneighbors indicando por linha os indices dos vizinhos mais proximos do padrao da linha
%   * n_c:				numero de clusters real (ground-truth)
%   * v_trueclass
%   * trueclass
%--------------------------------------------------------------------------------------------------
%% out:
% T:  matriz de adjacencia (esparsa) ja com k cortes da MST
% clusters: um vector linha com indice do cluster correspondente a cada amostra
% assocs: co-assocs esparsa
% verticesCortados: vertices cortados da MST obtida
%--------------------------------------------------------------------------------------------------

%% exemplo 
% para experimentar criar ensembles aleatorios
%  clear ensemble;
%  ns=100;
%  k=5;
%  for i=1:5
%     clusters=randint(1,ns,k+1)+1;
%     for j=1:k+1
%         I=find(clusters==j);
%         ensemble(i).nsamples_in_cluster(j)=length(I);
%         ensemble(i).clusters_m(j,1:length(I))=I;
%     end
%  end
%  clear clusters;clear I;
%  [out,clusters]=combina_generico2_sparseMST(ensemble,ns,[],[],3,[],[]);
%  gplot(out,dados,'r.-');


%% source:

% knn_mat: matriz nsXneighbors indicando por linha os indices dos vizinhos
% mais proximos do padrao da linha
if isempty(neighbors)
    knn_mat=[];neighbors=ns;
else
    knn_mat=zeros(ns,neighbors);  %lista ordenada por distancias crescentes: o primeiro e o mais proximo
    %falta fazer o processmanto da matriz, anteriormente:
%     for i = 1:ns
%         d(i,i)=inf;
%         [S,I] = sort(d(i,:));
%         knn_mat(i,:)=I(1:neighbors);
%     end
%     clear I;clear S;
end


%-------------------- Cria Co-assocs Sparse ---------------------------
n_clusterings=length(ensemble);
fprintf(1,'Ensemble N=%i\n',n_clusterings);

assocs=[];

for iter=1:n_clusterings;
   if ~(isnan(ensemble(iter).clusters_m))
       clusters_m=ensemble(iter).clusters_m; %clusters of current partition
       nsamples_in_cluster=ensemble(iter).nsamples_in_cluster;%number of samples in the clusters
       nclusters=length(nsamples_in_cluster); %number of clusters of partition
       
       % structures that will fill 
       dim=sum(nsamples_in_cluster.*(nsamples_in_cluster-1))/2; %number of elements of half distance matrix
       I=zeros(dim,1); %row index for sparse matrix
       J=zeros(dim,1); %column index for sparse matrix
       X = ones(dim, 1); %values of sparse matrix

       ntriplets = 0 ;
       

       for i=1:nclusters %for each cluster
           v= clusters_m(i,1:nsamples_in_cluster(i));% get all samples in cluster i
           if(~isempty(v))
               for j=1:nsamples_in_cluster(i) %for every sample j in cluster i
                   for k=j+1:nsamples_in_cluster(i) %for every sample k > j in cluster i
                           ntriplets = ntriplets + 1 ;
                           I(ntriplets) =  v(j);
                           J(ntriplets) =  v(k);
                   end
               end
           end
       end

       assocs_aux = sparse(I,J,X,ns,ns);% build square sparse matrix from half distance matrix
       if(iter==1)
           assocs=  assocs_aux;
       else
           assocs= assocs + assocs_aux;
       end
       %fprintf(1,'.');
   end
   
end
assocs= assocs + assocs' + speye(ns,ns).*n_clusterings;  %build full assoc matrix from half matrix
assocs=assocs/n_clusterings; %normalize matrix
fprintf(1,'\n');

%-------------------- normalizacao ---------------------------
[i,j,s]=find(assocs);%get row, col indices and values of nonzero elements
%%%%%%%%%%%%%%%%%%%%%
%(2b) para quando se converte em dissemelhanca nao ser convertido em zero
[IsMax]=find(s==1); % get indices of the max elements
%%%%%%%%%%%%%%%%%%%%%
%converter de semelhanca para distancia
s=1-s;
%%%%%%%%%%%%%%%%%%%%%
%(2b) para quando se converte em dissemelhanca nao ser convertido em zero
delta=eps; %eps returns the smallest value possible in used precision
% minimum distance vales (previous similarity max) become smallest possible non zero value
s(IsMax)=delta; 
%%%%%%%%%%%%%%%%%%%%%

%out.nassocs=assocs;
%build sparse coassoc from distances computed above
assocsout=sparse(i,j,s,ns,ns);

%-------------------- MST ---------------------------
%mst - pag 48
T = mst(assocsout);
%[i j v] = mst(A);
%mst(A, ['prim' | {'kruskal'})
% if ~exist([directo '\eac'],'dir')
%     mkdir([directo],['eac'])
% end
% cd([directo '\eac'])

%-------------------- visualizacao assocs --------------------
% figure(1)
% imagesc(assocs)
% colormap(jet)%colormap(gray)
% colorbar
% %print(gcf,'-depsc','-noui',[files_prefix 'assocs' '.eps']);
% %print(gcf,'-djpeg','-noui',[files_prefix 'assocs' '.jpg']);
% figure(2)
% imagesc(out)
% colormap(jet)%colormap(gray)
% colorbar

%-------------------------------------------------------------

% save([files_prefix 'nassoc.mat'],'nassocs');

%-------------------- extraccao da particao final ------------
%[out,clusters,verticesCortados]=extract_K_MST2(T,n_c,ns); %estou a usar o n_c: numero de clusters reais...
[T,clusters,verticesCortados]=extract_K_MST2(T,n_c,ns); %estou a usar o n_c: numero de clusters reais...

% figure(1);hold on;
% gplot(T,dados,'r.-');
