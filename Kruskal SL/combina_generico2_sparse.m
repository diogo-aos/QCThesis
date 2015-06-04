function [out]=combina_generico2_sparse(ensemble,ns,neighbors,methods,n_c,v_trueclass,trueclass)
%exemplo para experimentar a criacao das matrizes de co-assocs (ainda sem extraccao):
% clear ensemble;
% ns=100;
% k=5;
% for i=1:2
%     clusters=randint(1,ns,k+1)+1;
%     for j=1:k+1
%         I=find(clusters==j);
%         ensemble(i).nsamples_in_cluster(j)=length(I);
%         ensemble(i).clusters_m(j,1:length(I))=I;
%     end
% end
% clear clusters;clear I;
% [out]=combina_generico2_sparse(ensemble,ns,[],[],[],[],[]);

%inputs---------------------------------------------------------------------------------------------
%ensemble: 	estrutura do tipo struct que contem dois campos: clusters_m_ e nsamples_in_cluster_
%               cada um destes campos e uma cell para poder albergar
%               vectores de tamanho diference
%               (em vez de ler a lista de nomes vector_fich)
%ns: numero de instancias 
%neighbors:		matriz nsXneighbors indicando por linha os indices dos vizinhos mais proximos do padrao da linha
%methods: 		-------- metodos hierarquico usados de extracao da particao final
%					'single'   --- nearest distance
%       			'complete' --- furthest distance
%       			'average'  --- average distance
%       			'centroid' --- center of mass distance
%       			'ward'     --- inner squared distance
%					se method=[] entao nao faz extraccao de particao
                    % então single link é usado
%n_c:				numero de clusters real (ground-truth)
%v_trueclass
%trueclass
%--------------------------------------------------------------------------------------------------
%out:
%out(o).livre
%out(o).fixo
%.nsamples_in_cluster,
%.clusters_m
%.hit_counter
%--------------------------------------------------------------------------------------------------

% knn_mat: matriz nsXneighbors indicando por linha os indices dos vizinhos mais proximos do padrao da linha

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

n_clusterings=length(ensemble);
fprintf(1,'Ensemble N=%i\n',n_clusterings);

assocs=sparse(ns,neighbors);%zeros(ns,neighbors);

for iter=1:n_clusterings;
%    nsamples_in_cluster=[];
%    clusters_m=[];
   
   if ~(isnan(ensemble(iter).clusters_m))
       clusters_m=ensemble(iter).clusters_m;
       nsamples_in_cluster=ensemble(iter).nsamples_in_cluster;
       nclusters=length(nsamples_in_cluster);
       
       if isempty(knn_mat)
           
           dim=sum(nsamples_in_cluster.*(nsamples_in_cluster-1))/2;
           I=zeros(dim,1);
           J=zeros(dim,1);
           X = zeros (dim, 1) ;
           ntriplets = 0 ;
           for i=1:nclusters
               v= clusters_m(i,1:nsamples_in_cluster(i));
               if(~isempty(v))
                   for j=1:nsamples_in_cluster(i)
                       for k=j+1:nsamples_in_cluster(i)
                           ntriplets = ntriplets + 1 ;
                           I(ntriplets) =  v(j);
                           J(ntriplets) =  v(k);
                           X(ntriplets) = 1;
                       end
                   end
               end
           end
         
       else   %falta fazer - considerar apenas os neighbors...
           %todo
       end
       
       assocs_aux = sparse(I,J,X,ns,ns);
       if(iter==1)
           assocs=  assocs_aux;
       else
           assocs= assocs + assocs_aux;
       end
   end
end

assocs= assocs + assocs' + speye(ns,ns).*n_clusterings; 

%-------------------- normalizacao ---------------------------
% [i,j,s]=find(assocs);
% [v,I] = max(s);
% 
% %converter de semelhanca para distancia
% s=1-s/v;
% 
% %out.nassocs=assocs;
% out=sparse(i,j,s,ns,ns);

out.nassocs=assocs;

%-------------------- Cálculo da MST ------------------------
%mst - pag 48
%T = mst(A); %mst(A, ['prim' | {'kruskal'})

%-------------------- visualizacao assocs --------------------
% imagesc(assocs)
% colormap(jet)%colormap(gray)
% colorbar
% %print(gcf,'-depsc','-noui',[files_prefix 'assocs' '.eps']);
% %print(gcf,'-djpeg','-noui',[files_prefix 'assocs' '.jpg']);

% save([files_prefix 'assoc.mat'],'assocs');
%-------------------------------------------------------------


%-------------------- extraccao da particao final usando a MST------------
% [out,representante]=extract_K_MST(T,n_c); %estou a usar o n_c: numero de clusters reais...
%  
% clusters=zeros(1,ns);
% for k=1:length(representante)   
%     [d_i]= dfs(out,representante(k));
%     [I_di]=find(d_i~=-1);   %amostras pertencentes ao cluster k
%     %figure(1);plot(dados(I_di,1),dados(I_di,2),[color(k) style(k)])
%     clusters(I_di)=k;
% end

nassocs=full(assocs);

%-------------------- extraccao da particao final ------------
%para usar com qualquer método hierarquico para obtenção do cluster final.
if(~isempty(methods))
    for o=1:length(methods)
        method=methods{o};
        Z=apply_hierq2nassocs(nassocs,knn_mat,method);
        %k-livre
        [out.method(o).livre.nc_stable, out.method(o).livre.nsamples_in_cluster, out.method(o).livre.clusters_m ]= get_nc_stable_from_SL_dendro(Z, ns);
        out.method(o).livre.hit_counter=determine_ci(trueclass,out.method(o).livre.clusters_m,ns);
        %k-fixo
        [out.method(o).fixo.nsamples_in_cluster, out.method(o).fixo.clusters_m]= get_nc_clusters_from_SL_dendro(Z,n_c,[]);
        out.method(o).fixo.hit_counter=determine_ci(trueclass,out.method(o).fixo.clusters_m,ns);
    end
end
%-------------------------------------------------------------



