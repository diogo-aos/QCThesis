function [minkowski_measure,jacard_measure,purity,efficiency]=clustmeasure(clust,realClust)
% function [minkowski_measure,jacard_measure,purity,efficiency]=clustmeasure(clust,realClust)
% input: clust=vector with all the cluster # of each data point
%        realClust=vector ofthe starting place of each cluster 
%        (assuming the data points are sorted accordingly)

pNum=length(clust);
% S=the clutering result in pairs - S(i,j)=1 iff data point i and j are asigned to the same cluster
S=(repmat(clust,1,pNum)==repmat(clust',pNum,1));
for i=1:(length(realClust)-1);
    t(realClust(i):(realClust(i+1)-1))=i;
end
l=length(clust);
t(realClust(i+1):l)=i+1;
T=(repmat(t',1,l)==repmat(t,l,1));
% T=the true clutering (same definition as for S)


sum(sum(T==1));
sum(sum(T~=S));
minkowski_measure=sqrt(sum(sum(T~=S))/sum(sum(T==1)));
S1=S*2-1; % replace 0 for -1
TP=(T==S1); % all palces where both T and S equal 1
jacard_measure = sum(sum(TP))/(sum(sum(T~=S))+sum(sum(TP)));

efficiency = sum(sum(TP))/sum(sum(T==1)); % n11/(n10 + n11)
purity=      sum(sum(TP))/sum(sum(S==1)); % n11/(n01 + n11)



