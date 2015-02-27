function [ wX ] = pcaRW( x )
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
R = corr(x);

% get eigen*
[V,D]=eig(R);

%sort eigen
[D I] = sort(diag(D),'descend');
V = V(:,I); %already comes normalized

mX=x;
% projected data
pX = mX*V;

white=sqrt(D);
white=white.^(-1);
white=diag(white);

wX = pX * white;

%end of function
end

