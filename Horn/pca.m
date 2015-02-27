function [ V D u xp ] = pca( x )
  % Inputs:
  % x  : n x m data matrix with n dimensions and m points
  %
  % Outputs: [ V D u xp ]
  % V  : ordered principal components (each column is a component)
  % D  : ordered eigenvalues by descending order
  % u  : mean of data
  % xp : 

[n,d] = size(x);
mX = x - repmat(mean(x,1),n,1);

% covariance matrix
C = cov(x);

% get eigen*
[V,D]=eig(C);

%sort eigen
[D I] = sort(diag(D),'descend');
V = V(:,I); %already comes normalized

% projected data
xp = xc'*V;
xp = xp';

white=sqrt(latent)
white=white.^(-1)
white=diag(white)

wX = pX * white


%end of function
end