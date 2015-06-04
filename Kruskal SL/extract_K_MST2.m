function [S,clusters,verticesCortados]= extract_K_MST2(S,K,ns)
%-----------------
%in:
%S - grafo representante da MST (edges representam semelhancas)
%K - nº de clusters
%-----------------
%out:
%S - grafo representante da MST actualizado (edges representam semelhancas)
%clusters - vector linha com nº do cluster correspondente a cada amostra
%-----------------
%Nota: 
% - quando a MST tem mais de um edge com peso igual remove apenas o 1º
% - na versao extract_K_MST3- removem-se todos os edges o que pode produzir
%   um maior nº de clusters)
%-----------------

%antes de cortar verificar se o grafo tem sub-grafos independentes
tratados=[];
naotratados=[];

%d is the distance from vertex 1 to every other vertex on the MST
[d]= dfs(S,1);

%figure;plot(d)
%if the i-th distance is -1 it means the vertex chose above is not connected to the i-th vertex
[ii,jj,ss]=find(d==-1);

naotratados=ii; %set of vertices that are not connected to vertex 1 (chosen above)
%setdiff(A,B) returns the data in A that is not in B in sorted order
tratados=setdiff([1:ns],naotratados); %get set of samples that is connected to vertex 1
% tratados will hold all the different independent graphs in case they exist
grafosIndependentes=1;
if isempty(ii)
    clustersMST{grafosIndependentes}=[1:ns];
else % get all the unconnected subgraphs and store them in tratados
    clustersMST{grafosIndependentes}=tratados;
    while (~isempty(naotratados))
        grafosIndependentes=grafosIndependentes+1;
        [d]= dfs(S,naotratados(1));
        [ii,jj,ss]=find(d~=-1);
        tratados=[tratados ii'];
        clustersMST{grafosIndependentes}=ii';
        naotratados=setdiff([1:ns],tratados);
    end
end
%ver ismember or setdiff

%verificar se ja temos clusters suficientes
%se nao tivermos procurar a maior dissemelhanca em s e verificar em que
%conjunto é que ela pertence
%partir esse conjunto por essa ligacao

if grafosIndependentes==1
    inicio=1;
else
    inicio=grafosIndependentes;
end

% cut graph until the desired number of clusters is attained
% each cut generates 2 clusters, so K-1 cuts are required to have K clusters
% if there are independent graphs then less cuts are required = K-1-grafosIndependentes
% bigger edges are cut
verticesCortadosi=0;
for k=inicio:K-1 %so preciso de cortar KK-1 vezes (qdo se corta 1 ligacao geram-se dois clusters)
    [i,j,s]=find(S); %nonzero edges
    [v,I] = max(s); %maximum edge - biggest edge will be cut
    [Imaxs]=find(s==v); % check all the places where the maximum value exists
    Nmax=length(Imaxs); % check if more than one edge has maximum value
    if Nmax>2 % more than one edge has maximum value
        display(['Mais de 2 nós com semelhanca igual']);   %apenas para informacao    
    end
    %remove apenas a primeira aresta (independentemente de existirem mais)
    vertice1=i(I);vertice2=j(I);    %vertices do grafo cuja dissemelhanca é maior
    S(vertice1,vertice2)=0;S(vertice2,vertice1)=0;  %corta esse vertice
    verticesCortadosi=verticesCortadosi+1;verticesCortados(verticesCortadosi)=vertice1;
    verticesCortadosi=verticesCortadosi+1;verticesCortados(verticesCortadosi)=vertice2;

    % check to which subgraph the vertices belonged
    for hh=1:grafosIndependentes
        p(hh,1:2)=ismember([vertice1, vertice2],clustersMST{hh})
        if sum(p(hh,:))==2
            break
        end
    end

    [d_i]= dfs(S,vertice1); %d_i(i)=-1 if not reachable from vertice1
    [d_j]= dfs(S,vertice2); %d_j(i)=-1 if not reachable from vertice2
    clusterA=find(d_i~=-1); %get indices of vertices of newly formed cluster starting from vertice1
    clusterB=find(d_j~=-1); %get indices of vertices of newly formed cluster starting from vertice2
    clustersMST{hh}=clusterA; %add cluster A
    grafosIndependentes=grafosIndependentes+1;
    clustersMST{grafosIndependentes}=clusterB; % replace cluster that was split by cluter B
end

%format cluster for output
clusters=zeros(1,ns);
for k=1:length(clustersMST)
    %[d_i]= dfs(tt,representante(k));
    %[I_di]=find(d_i~=-1);   %amostras pertencentes ao cluster k
    %figure(1);plot(dados(I_di,1),dados(I_di,2),[color(k) style(k)])
    %clusters(I_di)=k;
    clusters(1,clustersMST{k})=k;
end
