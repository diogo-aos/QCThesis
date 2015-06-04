function [T, CostUnion,MSTreeEdgesUnion] =  MST_Kruskal2_test(assocs, clusters)
% The function takes CostMatrix as input and returns the minimum spanning tree T and cost of T
% Uses Kruskal's Algorithm
%-----------------
%Log: 
%Adaptado de GraphProject (MIT)
%6 de Out - versao 0.1 - retira apenas submatriz triangular superior de
%                        assocs (triu)
%14 de Out - versao 0.2 - qdo a arvore nao tem todos os elementos dava
%erro, tive de adaptar o ParentPointer e o TreeRank
%21 de Out - versao 0.2a - confirmar: 
%(MSTreeEdgesCounter < (n-1)) =>(MSTreeEdgesCounter <= (n-1))
%2 de Nov - saida dos MSTreeEdges
%5 de Nov
%acrescentei no fim if(MSTreeEdges==0) para o caso de assocs so ter
%singletons
%17 Nov
%-n= nnz(A); -> n=ns; (n = size (CostMatrix,1); %Number of vertices )?!?
%ainda n alterei mas devia considerar isto
%- calcular as spanning forests dos clusters passados como input
%-----------------
ns=size(assocs,1);
A=triu(assocs);

%n = size (CostMatrix,1); %Number of vertices
%n= nnz(A); %antes tinha
% n=ns; %varia dentro de cada cluster

%nova estrutura
MSTreeEdgesUnion=[];
CostUnion=0;

%para cada cluster
for k=1:length(clusters)
    clusts_cli=clusters{k};
    n= length(clusts_cli);
    Scli=A(clusts_cli,clusts_cli);  %sparse submatrix containg the edges in this cluster
    [i,j,s]=find(Scli);
    %as amostras reais referidas por i e j sao dadas por:
    %clusts_cli(i),clusts_cli(j)
    %se quiser criar uma matriz a partir destes valore terei de fazer:
    %nn_coassocs=sparse(clusts_cli(i),clusts_cli(j),s,nsamples,nsamples);
    
    % Extract the edge weights from the cost matrix
    % Sort the edges in a non decreasing order of weights
    % antes tinha:
    %[ii,jj,ss]=find(A);
    %EdgeWeights=[ss, ii, jj];
    EdgeWeights=[s, clusts_cli(i)', clusts_cli(j)'];
    
    SortedEdgeWeights = 0;
    SortedEdgeWeights = sortrows(EdgeWeights);
    %MATLAB: sorts the rows of the matrix X in ascending order as a group
    SortedEdgeWeights=flipud(SortedEdgeWeights); %descending order
    
    % First column of SortedEdgeWeights are the weights
    % Second and third column are the vertices that the edges connect
    m = size(SortedEdgeWeights,1); % number of edges
    %neste caso dentro do cluster
    
    % We use the Disjoint sets data structures to detect cycle while adding new
    % edges. Union by Rank with path compression is implemented here.
    
    % Assign parent pointers to each vertex. Initially each vertex points to
    % itself. Now we have a conceptual forest of n trees representing n disjoint
    % sets
    global ParentPointer ;
    ParentPointer = 0;
    %ParentPointer(1:n) = 1:n;
    %verify what are the present vertices
    % antes tinha:
    %I1=intersect([1:ns],SortedEdgeWeights(:,2));
    %I2=intersect([1:ns],SortedEdgeWeights(:,3));
    %U=union(I1,I2);
    %ParentPointer(U) = U;
    % agora so tenho os de cada cluster
    U= clusts_cli;
    ParentPointer(U) = U;
    
    % Assign a rank to each vertex (root of each tree). Initially all vertices
    % have the rank zero.
    TreeRank = 0;
    %TreeRank(1:n) = 0;
    TreeRank(U) = 0;
    
    % Visit each edge in the sorted edges array
    % If the two end vertices of the edge are in different sets (no cycle), add
    % the edge to the set of edges in minimum spanning tree
    MSTreeEdges = [];
    MSTreeEdgesCounter = 0; i = 1;
    while ((MSTreeEdgesCounter <= (n-1)) && (i<=m))
        %while ((MSTreeEdgesCounter < (n-1)) && (i<=m))
        %     Find the roots of the trees that the selected edge's two vertices
        %     belong to. Also perform path compression.
        root1=0; root2=0; temproot=0;
        temproot = SortedEdgeWeights(i,2);
        root1 = FIND_PathCompression(temproot);
        
        temproot = SortedEdgeWeights(i,3);
        root2 = FIND_PathCompression(temproot);
        
        if (root1 ~= root2)
            MSTreeEdgesCounter = MSTreeEdgesCounter + 1;
            MSTreeEdges(MSTreeEdgesCounter,1:3) = SortedEdgeWeights(i,:);
            if (TreeRank(root1)>TreeRank(root2))
                ParentPointer(root2)=root1;
            else
                if (TreeRank(root1)==TreeRank(root2))
                    TreeRank(root2)=TreeRank(root2) + 1;
                end
                ParentPointer(root1)=root2;
            end
        end
        i = i + 1;
    end
    
    if ~isempty(MSTreeEdges)
        cost = sum (MSTreeEdges(:,1));
        CostUnion= cost + CostUnion;
    end
    
    MSTreeEdgesUnion=[MSTreeEdgesUnion; MSTreeEdges];
end

% MSTreeEdgesCounter = 0;
% T = 0;
% T(1:n,1:n)=0;
% while (MSTreeEdgesCounter < (n-1))
%     MSTreeEdgesCounter = MSTreeEdgesCounter + 1;
%     T(MSTreeEdges(MSTreeEdgesCounter,2),MSTreeEdges(MSTreeEdgesCounter,3))=1;
%     T(MSTreeEdges(MSTreeEdgesCounter,3),MSTreeEdges(MSTreeEdgesCounter,2))=1;
% end
if(MSTreeEdges==0)
    [ii,jj,ss]=find(A);
    T=sparse(ii,jj,ss,ns,ns);
else
    %antes:
    %T=sparse(MSTreeEdges(:,2),MSTreeEdges(:,3),MSTreeEdges(:,1),ns,ns);
    %agora:
    T=sparse(MSTreeEdgesUnion(:,2),MSTreeEdgesUnion(:,3),MSTreeEdgesUnion(:,1),ns,ns);
    T=T+T';
end
%T
%cost

