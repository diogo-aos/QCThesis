% function [adj_l, adj_l_cost, Xmst,XmstValue] = mste_new_s(S,X,ObjLab)
% 
% Minimal or Minimum Spanning Tree based on similarity matrix
% MST in short: use S(sparse nxn) to form (n-1) lines to connect (n)
% objects (in X (n x p)) in the shortest possible way in the (p)
% dimensional variable-space, under the condition 'no closed loops
% allowed'. Uses Prim Algorithm
% 
% in: S (objects x objects) Adjacency matrix (with similarities) 
%     (D = squareform(pdist(X,'euclidean'));) (S = max(max(D))-D;)
%     X (objects x n) data-table
%     ObjLab (objects x 1) object labels for plotting
%
% out:Xmst (objects-1 x 2) link set between 'objects' indexed as rows in X
%     D (objects x objects) distance matrix
%     adj_l ({object x adjacent_objects_in_MST}) lista de adjacencias na MST 
%                             para cada amostra, implementada como cell {}
%     adj_l_cost ({object x cost_to_adjacent_objects_in_MST}) lista de custos 
%                            (dist. euclideana) associados a lista de adjacencias na MST 
%                            para cada amostra, implementada como cell {}
%
% based on: F.S.Hillier and G.J.Lieberman 'Introduction to operational research' 7(2001)415-420
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Altered by Ana Fred 11/05/2006
% Altered by ALourenco 15/04/2009
% Altered by ALourenco 14/09/2009
% Altered by ALourenco 14/10/2009
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [T,cost] = mste_new_s2(S,X,ObjLab)

if (nargin == 0)
   help mst_new
   return
end   

[nX,mX] = size(S);
if (mX < 2)
    error('ERROR: "S" must contain at least three objects to compute MST');
    %tlz nao faça sentido este comentario... (mais de dois ... nao?)
end

%Dmax = max(max(triu(D,1)))*10;
%Xmst=zeros(nX-1,2);
XmstValue=zeros(nX-1,1);

%lista de adjacencias...
clear adj_l adj_l_cost
for a=1:nX
    adj_l{a}=[];adj_l_cost{a}=[];
end
%temp = randperm(nX); % starting with random node will lead to the same solution!
%temp = temp(1);
%temp1 = setdiff(1:nX,temp);
%[Dmin,Dwin] = min(D(temp,temp1));
%Xmst(1,:) = [temp temp1(Dwin)];

%começando pelo vertice 1 - procurar a ligação de menor custo
% no nosso caso de maior semelhanca
%[Dmin,Dwin] = min(D(1,2:nX));   %dist do 1º no a todos os outros...
[Dmin,Dwin] = max(S(1,2:nX));   %dist do 1º no a todos os outros...
Xmst(1,:) = [1 Dwin+1];         %o 1º vertice é o q esta em 1º (Dwin+1 pq 2:nX p compensar o 1)
%Xmst - conjunto dos elemento ja inseridos na MST (em cada linha os vertices ligados)
XmstValue(1)=Dmin;

adj_l{1}=[Dwin+1]; 
%adj_l - lista de adjacencias da arvore minima de suporte
% cell com nX elementos que diz que vertices estam ligados (apenas pares de vertices)
% neste caso o vertice 1 está ligado ao vertice Dwin+1, logo adj_l{Dwin+1}=[1] e adj_l{1}=[Dwin+1] 
adj_l_cost{1}=[Dmin]; 
adj_l{Dwin+1}=[1]; 
adj_l_cost{Dwin+1}=[Dmin]; 

%PRIM's algorithm
% start with any vertex as a single-vertex MST, then add V-1 edges to it,
% always taking next a minimal edge that connects a vertex on the MST to a
% vertex not yet on the MST
for a=2:nX-1
    mindist = 0;%Dmax;
    Xmstlist = unique(Xmst(:))';            %tira as repeticoes
    Xmstnotlist = setdiff(1:nX,Xmstlist);   %retorna os elementos em 1:nX que nao estao em Xmstlist
    for aa=Xmstlist %percorre a lista dos q ja estam na MST
        %[Dmin,Dwin] = min(D(aa,Xmstnotlist)); %encontra o minimo entre a actual MST e Xmstnotlist
        [Dmin,Dwin] = max(S(aa,Xmstnotlist)); %encontra o max entre a actual MST e Xmstnotlist
        %if (Dmin < mindist)
        if (Dmin > mindist)
            minindex = [aa Xmstnotlist(Dwin)];
            mindist = Dmin;
        end
    end
    Xmst(a,:) = minindex;
    XmstValue(a)=mindist;
    adj_l{minindex(1)}=[adj_l{minindex(1)} minindex(2)]; 
    adj_l_cost{minindex(1)}=[adj_l_cost{minindex(1)} mindist];
    adj_l{minindex(2)}=[adj_l{minindex(2)} minindex(1)]; 
    adj_l_cost{minindex(2)}=[adj_l_cost{minindex(2)} mindist];
end

T=sparse(Xmst(:,1),Xmst(:,2),XmstValue,nX,nX);
T=T+T';
cost = sum (XmstValue);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%plot da MST sobre os dados
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin > 1
    nO = size(X,1);
    if (nargin == 3)
        if (nO ~= nX)
            error('ERROR: number of labels in "ObjLab" must be the same as number of objects in "X"');
        end
    end
    if (size(X,2) == 2)
        plot(X(:,1),X(:,2),'.r')
        if (nargin == 3)
            text(X(:,1),X(:,2),ObjLab);
%         else
%             text(X(:,1),X(:,2),num2str([1:nX]'));
        end
        grid
        hold on
        for a=1:nX-1
            plot([X(Xmst(a,1),1) X(Xmst(a,2),1)],[X(Xmst(a,1),2) X(Xmst(a,2),2)]);
        end
        hold off
        xlabel('D-one');
        ylabel('D-two');
        title('Minimum Spanning Three (MST) for data table (based on similarities)');
    elseif (size(X,2) == 3)
        plot3(X(:,1),X(:,2),X(:,3),'.r')
        if (nargin == 3)
            text(X(:,1),X(:,2),X(:,3),ObjLab);
%         else
%             text(X(:,1),X(:,2),X(:,3),num2str([1:nX]'));
        end
        grid
        hold on
        for a=1:nX-1
            plot3([X(Xmst(a,1),1) X(Xmst(a,2),1)],[X(Xmst(a,1),2) X(Xmst(a,2),2)],[X(Xmst(a,1),3) X(Xmst(a,2),3)]);
        end
        hold off
        xlabel('D-one');
        ylabel('D-two');
        zlabel('D-three');
        title('Minimum Spanning Three (MST) for data table');
    else
        disp('WARNING: plotting only possible for two or three variables in "X"');
    end
end
