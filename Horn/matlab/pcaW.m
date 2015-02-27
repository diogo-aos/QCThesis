function [ wX latent pc] = pcaW( x )
  % Inputs:
  % x  : n x m data matrix with n points and m dimensions
  %
  % Outputs: [ V D u xp ]
  % wx : projected data

[n,d]=size(x)

mX = x - repmat(mean(x,1),n,1)

%[pc,score,latent,tsquare] = princomp(mX);

%C = cov(mX);
C = mX' * mX;
C = C ./ n;

% get eigen*
[V,D]=eig(C);

%sort eigen
[D I] = sort(diag(D),'descend');
V = V(:,I); %already comes normalized

pc=V;
latent=D;

pX = mX * pc;

white=sqrt(latent);
white=white.^(-1);
white=diag(white);

wX = pX * white;

%end of function
end

