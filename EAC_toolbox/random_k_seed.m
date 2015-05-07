function centroids_seed=random_k_seed(k,nsamples)


%inicializa aleatoriamente os centroides em 'k' indices distintos seleccionados 
%entre 1 e 'nsamples'


%Last update: 1/Oct/2001

   centroids_seed=[random_num(nsamples)];
   while (length(centroids_seed)<k)
      nn=random_num(nsamples);
      if isempty(find(centroids_seed==nn))
         centroids_seed=[centroids_seed nn];
      end
   end