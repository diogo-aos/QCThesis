%plot clusters


x=zeros(length(xyData),max(clust));
x=repmat(1:max(clust),length(xyData),1)';
index=repmat(clust,1,max(clust))';
x=(x==index);
x(:,realClust)=x(:,realClust)+2;
BWcmap=[1 1 1;0 0 0;1 0 0;0.7 0.7 0.7;0.2 0.2 0.2];

colormap(BWcmap);
image(x*2)
