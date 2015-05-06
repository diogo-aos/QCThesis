function clus1=converte_indices_em_pos(clusts1,nsamples)

%converte o numero das amostras na matriz dos clusters em indices de posicoes nao nulas
%[1 2 5 0 0] -> [1 1 0 0 1 0]

[nlines1,cols1]=size(clusts1);
clus1=zeros(nlines1,nsamples);
for i=1:nlines1
   first_zero_pos=find(clusts1(i,:)==0);
   if isempty(first_zero_pos)
      col=cols1;
   else
      col=(first_zero_pos(1)-1);
   end
   for j=1:col
      clus1(i,clusts1(i,j))=1;
   end
end
         