function [ V D u xp ] = pca( x )
  % Inputs:
  % x  : n x m data matrix with n dimensions and m points
  %
  % Outputs: [ V D u xp ]
  % V  : ordered principal components (each column is a component)
  % D  : ordered eigenvalues by descending order
  % u  : mean of data
  % xp : 

sizeOfX=size(x);
dim=sizeOfX(1);
points=sizeOfX(2);

% compute mean
u = mean(x');

% center data
for i=1:dim
    xc(i,:)=x(i,:)-u(1);
end

% covariance matrix
C = (xc*xc') / points;

% get eigen*
[V,D]=eig(C);

%sort eigen
[D I] = sort(diag(V),'descend');
V = V(:,I); %already comes normalized

% projected data
xp = xc'*V;
xp = xp';

%end of function
end

