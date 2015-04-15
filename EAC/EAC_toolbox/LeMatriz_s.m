function a=LeMatriz_s(fich)

if isempty(fich)
   fich='g:\Temp\amostras.txt';
end


fp=fopen(fich,'rt');
%LEITURA DE NUMERO DE AMOSTRAS E DIMENSAO DO VECTOR
ss=fgets(fp);  %le primeira linha para string
ll=sscanf(ss,'%e');
n=ll(2);
d=ll(1);

%LEITURA DAS AMOSTRAS
a= zeros(n,d);
for i=1:n
   ss=fgets(fp);
   ll=sscanf(ss,'%e');
   for j=1:d
	a(i,j)=ll(j);
   end
end

fclose(fp);

%plotmatrix(a);
%set(gcf,'color','w')

%figure
%set(gcf,'color','w')
%plot(a(:,1),a(:,2),'k.')
