function hit_counter=determine_ci (clusts1,clusts2,nsamples)

%determina quais são os clusters em correspondencia analisando o numero de amostras comuns
%Utiliza a interseccao entre histogramas para medir a semelhance entre os clusters


[nlines1,cols1]=size(clusts1);
[nlines2,cols2]=size(clusts2);
nls1=nlines1;
nls2=nlines2;
hit_counter=0;  %contador de amostras coincidentes nos mesmos clusters
hit_aux=0;

%converte o numero das amostras na matriz dos clusters em indices de posicoes nao nulas
%[1 2 5 0 0] -> [1 1 0 0 1 0]
cluso1=converte_indices_em_pos(clusts1,nsamples);
cluso2=converte_indices_em_pos(clusts2,nsamples);
clus1=cluso1;
clus2=cluso2;
for nc=1:nlines2
   max_intersection=0;
   hit_aux=0;
   ind_c1=0;
   ind_c2=0;
   if (nls1 <= 0) | (nls2 <= 0)% o segundo conjunto tem mais clusters
      break
   else
      for i=1:nls1 %determina o melhor match
         for j=1:nls2
            %             hh=hist_intersection(clus1(i,:),clus2(j,:)); %conta co-ocorrencias de amostras no cluster
            hh=clus1(i,:)*clus2(j,:)';  %é o mesmo qu acima mas mais efficiente
            %            hist_int=hh/max(sum(clus1(i,:)),sum(clus2(j,:)));
            hist_int=hh/(clus1(i,:)*clus1(i,:)'+clus2(j,:)*clus2(j,:)'-clus1(i,:)*clus2(j,:)');
            if hist_int > max_intersection
               max_intersection=hist_int;
               hit_aux=hh;
               ind_c1=i;
               ind_c2=j;
            end
         end
      end
      if (ind_c1==0)& (ind_c2==0)
         break
      else
         clus1=remove_line(clus1,ind_c1);
         clus2=remove_line(clus2,ind_c2);
         nls1=nls1-1;
         nls2=nls2-1;
         hit_counter=hit_counter+hit_aux/nsamples;
      end
   end
end


