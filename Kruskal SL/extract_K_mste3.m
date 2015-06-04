function [S,clustersMST]= extract_K_mste3(S,K,ns,MSTreeEdges)
%-----------------
%in:
%S - grafo representante da MST (edges representam semelhancas)
%K - nº de clusters
%ns - nº de amostras
%MSTreeEdges - grafo representante da MST, 1ª coluna c pesos; ordenado decrescentemente  (maior semelhanca 1º)
%-----------------
%out:
%S - grafo representante da MST actualizado (edges representam semelhancas)
%clustersMST - cell com clusters
%OLD- clusters - vector linha com nº do cluster correspondente a cada amostra

%-----------------
%Nota: 
% - quando a MST tem mais de um edge com peso igual remove apenas o 1º
% - Adaptacao de extract_K_MST2 p matrizes de semelhanca
% - grafo de entrada (S) uma matriz de adjacencias com as semelhanca (->criterio de corte "min" em vez de "max")
% - usa DFS do BGL para verificar conectividade entre vertices
%-----------------
%Log: 
%30 de Set - criterio de corte "max" (o resto tudo igual)
%13 de Out - novos parametros de saida ( cell c clusters)
%2 de Nov - novo parametro de entrada (MSTreeEdges - criado aquando da
%construcao da MST) que pode ser usado para escolher quais os nos a cortar
%----------------- 



%antes de cortar verificar se o grafo tem sub-grafos independentes
tratados=[];naotratados=[];
[d]= dfs(S,1);
%figure;plot(d)
[ii,jj,ss]=find(d==-1);
naotratados=ii;
tratados=setdiff([1:ns],naotratados);
grafosIndependentes=1;
if isempty(ii)
    clustersMST{grafosIndependentes}=[1:ns];
else
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
verticesCortadosi=0;
for k=inicio:K-1 %so preciso de cortar KK-1 vezes (qdo se corta 1 ligacao geram-se dois clusters)
    vertice1=MSTreeEdges(end-verticesCortadosi,2);
    vertice2=MSTreeEdges(end-verticesCortadosi,3);
    verticesCortadosi=verticesCortadosi+1;
    S(vertice1,vertice2)=0;S(vertice2,vertice1)=0;  %corta esse vertice
    for hh=1:grafosIndependentes
        p(hh,1:2)=ismember([vertice1, vertice2],clustersMST{hh});
        if sum(p(hh,:))==2
            break
        end
    end
    [d_i]= dfs(S,vertice1); %d_i(i)=-1 if not reachable from vertice1
    [d_j]= dfs(S,vertice2); %d_j(i)=-1 if not reachable from vertice2
    clusterA=find(d_i~=-1);
    clusterB=find(d_j~=-1);
    clustersMST{hh}=clusterA;
    grafosIndependentes=grafosIndependentes+1;
    clustersMST{grafosIndependentes}=clusterB;
end


