function Z=apply_hierq2nassocs1(nassocs,method)

% APLICA O ALGORITMO em method DO MATLAB PARA CONSTRUIR O DENDROGRAMA A PARTIR DA MATRIZ DE NASSOCS
%method='single'   --- nearest distance
%       'complete' --- furthest distance
%       'average'  --- average distance
%       'centroid' --- center of mass distance
%       'ward'     --- inner squared distance
%se method=[] então single link é usado

Z=[];
[m, n] = size(nassocs);
p = (m-1):-1:2;
I = zeros(m*(m-1)/2,1);
I(cumsum([1 p])) = 1;
I = cumsum(I);
I = (I-1)*m;	%0 0 0 ... 0 m m m ... m 2*m ... 2*m ... (m-2)*m 
J = ones(m*(m-1)/2,1);
J(cumsum(p)+1) = 2-p;
J(1)=2;
J = cumsum(J);	%2 3 4 ... m 3 4 5 ... m 4 5 .... m ...  m-1
Y=zeros(m,m);
%similarity - larger its value, closer or more alike patterns are


Y=nassocs;

%convert to dissimilarity
Y=max(max(Y))-Y;

if isempty(method)
   method='single'
end

%linkage use an input in pdist format (that's one line with the triangular lower part of the dissimilarity matrix)
Z=linkage(Y(J+I)',method);	%Z has hierarchical cluster information (size m-1 by 3)
