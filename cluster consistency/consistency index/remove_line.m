function clus1=remove_line(clus1,ind_c1)

[lines,cols]=size(clus1);

for i=ind_c1:lines-1
   clus1(i,:)=clus1(i+1,:);
end
clus1(lines,:)=zeros(1,cols);

