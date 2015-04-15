function [nc_stable, ns_in_cl, clusts ]= get_nc_stable_from_SL_dendro(Z,ns)

%determina o cluster de maior lifetime

%finding the maximum lifetime jump on the dendrogram
dif=Z(2:end,3)-Z(1:end-1,3);
[maximo,indice]=max(dif);

indice=Z(find(Z(:,3)>Z(indice,3)),3);
if isempty(indice)
   cont=1;
else
   cont=length(indice)+1;
end

th=maximo;

%testing the situation when only 1 cluster is present
%max>2*min_interval -> nc=1
minimo=min(dif(find(dif~=0)));
if maximo<2*minimo
   cont=1;
end
%fprintf(1,'maximo: %f\nminimo: %f\n2*minimo: %f\nnc_stable: %f\n',maximo,minimo,2*minimo,cont);

nc_stable=cont

if nc_stable > 1
	[H,clusters] = dendrogram(Z,nc_stable);
	clusts=[];
	for k = 1 : nc_stable
   	a=find(clusters==k);
	   ns_in_cl(k)=length(a);
   	clusts(k,1:ns_in_cl(k))=a(1:ns_in_cl(k))';
   end
else	%ns_stable=1
   ns_in_cl=ns;
   clusts=1:ns;
end
