% This script demonstare the QC alogorithm in a truncated SVD space 
% on microarray data the run time should take several minutes

% load the data matrix
load spellman-demo
% perform SVD - the result are 3 matrixes s.t  genes x S x samples = M
[genes,S,samples] = svd(M,0);
dims=4;
q=2.4; % q=1/(2*sigma^2) => sigma=0.46 (smaller q -> less clusters)

%xyData=samples(:,1:dims); % load dims most significant vectors to xyData
xyData=genes(:,1:dims); % load dims most significant vectors to xyData

% data normalization (gives all vector unit length)
n = normc(xyData');
xyData=n';


%show_qc; % run qc and then plot the result (if more than 2 dimentions are used -
         % then the result is the projection of on the first to dimentions)
         % this procedure does not perform the gradient descent and uses only
         % for presentation purpose 

% QC
D=graddesc(xyData,q,80);  %performs gradient descent on xyData with 20 steps
clust=FineCluster(D,0.1); % "collapse" the points to their final places and
                          % return the division of data into clusters  

plotClust;
[mm jm purity efficiency]=clustmeasure(clust,realClust); %minkowski measure and pairwise measure
QCjacard_measure=jm;
QCminkowski_measure=mm;
title(strcat('QC clustering ',int2str(dims),' dimensions  jacard=' ,num2str(QCjacard_measure) ))