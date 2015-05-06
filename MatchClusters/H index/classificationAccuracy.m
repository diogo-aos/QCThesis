function [accuracy]=classificationAccuracy(gt,clustering)
% gt = ground truth
% clustering = clustering to check accuracy

% rows should be clusters, which mean horizontal rectangle matrices
if(size(gt,2)<size(gt,1))
  gt=gt';
end
if(size(clustering,2)<size(clustering,1))
  clustering=clustering';
end

k=max([gt clustering]); % maximum number of clusters
n=numel(clustering); % number of samples

M1=zeros(n,k);
M1(sub2ind(size(M1),1:n,gt))=1; % incidence matrix for gt
%clustering=clustering+1;
M2=zeros(n,k);
M2(sub2ind(size(M2),1:n,clustering))=1; % incidence/binary matrix for clustering


CE=M1'*M2; % KxK matrix of common clusterings
[~,accuracy]=munkres(-CE);
%cost=cost+(gt==1)'*(clustering==2 | clustering;
accuracy=(-accuracy)/length(gt);
end