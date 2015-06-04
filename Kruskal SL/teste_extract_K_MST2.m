%Teste para confirmar a implementacao do extract_K_MST2 usando dados
%benchmark

clear;clc;

data_name{1}='cigar';
data_name{2}='spiral';
data_name{3}='iris';
data_name{4}='breast_cancer';
data_name{5}='barras_c3';
data_name{6}='barras_c3a';
data_name{7}='halfrings2_arl';
data_name{8}='halfrings1_arl';  %esta a dar diferente (no fich2 - rever)
data_name{9}='r_2_new';
data_name{10}='rings';
data_name{11}='d1';
data_name{12}='d2';
data_name{13}='d3';
data_name{14}='log_yeast_cellcycle';
data_name{15}='std_yeast_cellcycle';
data_name{16}='optdigits_r_tra_1000';
data_name{17}='img_complex';
data_name{18}='img_complex2';
data_name{19}='img_complex3';
data_name{20}='image_1_Martin';
data_name{21}='image_2_Martin';
data_name{22}='image_3_Martin'; %esta a dar diferente (no fich2 - rever)
data_name{23}='textures';   
data_name{24}='crabs'; %esta a dar diferente (no fich2 - rever)
data_name{25}='crabs_norm'; %esta a dar diferente (no fich2 - rever)
data_name{26}='house_votes';
data_name{27}='house_votes_norm';
data_name{28}='ionosphere';
data_name{29}='ionosphere_norm'; %esta a dar diferente (no fich2 - rever)
data_name{30}='pima';
data_name{31}='pima_norm';
data_name{32}='wine'; %esta a dar diferente (no fich2 - rever)
data_name{33}='wine_norm';

%path_base='E:\toolbox\Databases\'   %IST
%path_base='E:\clustering\evaclue\joaoDuarte\Code\Databases\' %portatil
path_base='C:\clustering\evaClue\joaoduarte\Code\Databases\' %IST-novo
N_ensemble=150;
methods={'single'}
methods_acronym={'SL','MST'}
set(0,'RecursionLimit',1000)

%%
for n=1:length(data_name)
    clear dados;
    dados=mydados(data_name{n},path_base);
    
    path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
%     fich1=['CE_K_Var_' dados.nome '_N=150_resu_struct.mykmeans']
%     fich2=['CE_' dados.nome '_N=150_resu_struct.mykmeans'];
%     fich1a=['CE_K_Var_sqrt_' dados.nome '_N=150_resu_struct.mykmeans']
%     fich2a=['CE_K_Var_linear_' dados.nome '_N=150_resu_struct.mykmeans']
    fichsProc={['CE_K_Var_' dados.nome '_N=150_resu_struct.mykmeans'],...
               ['CE_' dados.nome '_N=150_resu_struct.mykmeans']}
%                ['CE_K_Var_sqrt_' dados.nome '_N=150_resu_struct.mykmeans'],...
%                ['CE_K_Var_linear_' dados.nome '_N=150_resu_struct.mykmeans']}
    %------------------------------------
    %caracteristicas do data-set
    fich_dados=[path_base dados.nome '.txt'];
    dados.data=leMatriz_s(fich_dados);
    [ns,dimensoes]=size(dados.data);
    %--------- groud truth ---------
    v_trueclass=dados.fim-dados.inicio+1;
    trueclass=zeros(length(v_trueclass),max(v_trueclass));
    for i=1:length(v_trueclass)
        trueclass(i,1:v_trueclass(i))=[dados.inicio(i):dados.fim(i)];
    end
           
    for ff=1:2
    
    
    %----------------------------------------------------------------------
    %combina usando metodo classico
    clear EAC;clear CE;
    filename=[path_fich fichsProc{ff}];
    load(filename,'-mat'); %load da matriz de coassocs
    
    co_assocs= exp_1.CE.co_assocs + exp_1.CE.co_assocs' + speye(ns,ns).*exp_1.CE.co_assocs_norm_fact; 
    co_assocs=co_assocs/exp_1.CE.co_assocs_norm_fact;

    co_assocs_f=full(co_assocs); %full para metodo classico
    for o=1:length(methods)
        method=methods{o};
        Z=apply_hierq2nassocs(co_assocs_f,[],method);
        nsamples_in_cluster=[];clusters_m=[];
        %k-fixo
        [nsamples_in_cluster,clusters_m]= get_nc_clusters_from_SL_dendro(Z,dados.nc,[]);
        hit_counter=determine_ci(trueclass,clusters_m,ns);
        EAC.(methods_acronym{o}).comb_partition_fixedk.nsamples_in_cluster=nsamples_in_cluster;
        EAC.(methods_acronym{o}).comb_partition_fixedk.clusters_m=clusters_m;
        EAC.(methods_acronym{o}).comb_partition_fixedk.nc=dados.nc;
        EAC.(methods_acronym{o}).comb_partition_fixedk.ci=hit_counter;
        resumo(n,o,ff)=hit_counter;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % extrai usando a mst
    [IsMax]=find(co_assocs==1);
    co_assocs=1-co_assocs;
    delta=eps;
    co_assocs(IsMax)=delta;
    [i,j,s]=find(co_assocs);
    %s=1-s;
    co_assocs=sparse(i,j,s,ns,ns);
    acr='MST';
    %if n~=29&&n~=6&&n~=8&&n~=30
        T = mst(co_assocs);
        %[out,clusters]=extract_K_MST2(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
        [out,clusters]=extract_K_MST3(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
        %for k=1:dados.nc
        for k=1:max(clusters)
            I=find(clusters==k);
            EAC.(acr).comb_partition_fixedk.nsamples_in_cluster(k)=length(I);
            EAC.(acr).comb_partition_fixedk.clusters_m(k,1:length(I))=I;
        end
        EAC.(acr).comb_partition_fixedk.ci=determine_ci(trueclass,EAC.(acr).comb_partition_fixedk.clusters_m,ns);
        resumo(n,length(methods)+1,ff)=EAC.(acr).comb_partition_fixedk.ci;
    %else
    %    EAC.(acr).comb_partition_fixedk.ci=NaN;
    %    resumo2(n,6)=NaN;
    %end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %save(fich_gravar,'-mat','-append','EAC');
    end
end
save('teste_extract_K_MST3.mat','resumo');

%%
%Verificacao dos casos onde existe diferencas
clear dados;
% %1)
% n=16;
% dados=mydados(data_name{n},path_base);
% path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
% fich=['CE_K_Var_' dados.nome '_N=150_resu_struct.mykmeans']

%2)
n=5;
dados=mydados(data_name{n},path_base);
path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
fich=['CE_' dados.nome '_N=150_resu_struct.mykmeans']

filename=[path_fich fich];
fich_dados=[path_base dados.nome '.txt'];
dados.data=leMatriz_s(fich_dados);
[ns,dimensoes]=size(dados.data);
%--------- groud truth ---------
v_trueclass=dados.fim-dados.inicio+1;
trueclass=zeros(length(v_trueclass),max(v_trueclass));
for i=1:length(v_trueclass)
    trueclass(i,1:v_trueclass(i))=[dados.inicio(i):dados.fim(i)];
end
clear EAC;clear CE;
filename=[path_fich fich];
load(filename,'-mat'); %load da matriz de coassocs

co_assocs= exp_1.CE.co_assocs + exp_1.CE.co_assocs' + speye(ns,ns).*exp_1.CE.co_assocs_norm_fact;
co_assocs=co_assocs/exp_1.CE.co_assocs_norm_fact;

co_assocs_f=full(co_assocs); %full para metodo classico
methods={'single'};methods_acronym={'SL'}
for o=1:length(methods)
    method=methods{o};
    Z=apply_hierq2nassocs(co_assocs_f,[],method);
    nsamples_in_cluster=[];clusters_m=[];
    %k-fixo
    [nsamples_in_cluster,clusters_m]= get_nc_clusters_from_SL_dendro(Z,dados.nc,[]);
    hit_counter=determine_ci(trueclass,clusters_m,ns);
    EAC.(methods_acronym{o}).comb_partition_fixedk.nsamples_in_cluster=nsamples_in_cluster;
    EAC.(methods_acronym{o}).comb_partition_fixedk.clusters_m=clusters_m;
    EAC.(methods_acronym{o}).comb_partition_fixedk.nc=dados.nc;
    EAC.(methods_acronym{o}).comb_partition_fixedk.ci=hit_counter;
    resumo_n(n,o)=hit_counter;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% extrai usando a mst

%%
%fazer conv para dessemelhanca antes de converter para esparsa
delta=eps;
[IsMax]=find(co_assocs_f==1);
co_assocs_d=1-co_assocs_f;
co_assocs_d(IsMax)=delta;
[i,j,s]=find(co_assocs_d);
%s=1-s;
co_assocs=sparse(i,j,s,ns,ns);
figure;imagesc(co_assocs)
acr='MST';
%if n~=29&&n~=6&&n~=8&&n~=30
T = mst(co_assocs);
figure;imagesc(T);colorbar
%[out,clusters]=extract_K_MST2(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
[out,clusters]=extract_K_MST3(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
for k=1:dados.nc
    I=find(clusters==k);
    EAC.(acr).comb_partition_fixedk.nsamples_in_cluster(k)=length(I);
    EAC.(acr).comb_partition_fixedk.clusters_m(k,1:length(I))=I;
end
EAC.(acr).comb_partition_fixedk.ci=determine_ci(trueclass,EAC.(acr).comb_partition_fixedk.clusters_m,ns);
resumo_n(n,length(methods)+1)=EAC.(acr).comb_partition_fixedk.ci;


EAC.SL.comb_partition_fixedk.nsamples_in_cluster
EAC.MST.comb_partition_fixedk.nsamples_in_cluster

figure;imagesc(co_assocs);colorbar
figure;spy(co_assocs)

size(dados.data)
figure;hold on
plot(dados.data(:,1),dados.data(:,2),'*')
gplot(T,dados.data,'r.-');
plot(dados.data(unij,1),dados.data(unij,2),'pk')


[i,j,s]=find(T);
[out,clusters]=extract_K_MST2(T,dados.nc,ns)

%% teste extract3 (casos patologicos)
clear dados;close all
%a)
% n=10;
% dados=mydados(data_name{n},path_base);
% path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
% fich=['CE_K_Var_' dados.nome '_N=150_resu_struct.mykmeans']
%b)
% n=22;
% dados=mydados(data_name{n},path_base);
% path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
% fich=['CE_' dados.nome '_N=150_resu_struct.mykmeans']
%b)
n=14;
dados=mydados(data_name{n},path_base);
path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
fich=['CE_' dados.nome '_N=150_resu_struct.mykmeans']
% %c)
% n=29;
% dados=mydados(data_name{n},path_base);
% path_fich=[path_base dados.nome '\data_CE\CE_N=150\'];
% fich=['CE_' dados.nome '_N=150_resu_struct.mykmeans']
%
filename=[path_fich fich];
fich_dados=[path_base dados.nome '.txt'];
dados.data=leMatriz_s(fich_dados);
[ns,dimensoes]=size(dados.data);

%--------- groud truth ---------
v_trueclass=dados.fim-dados.inicio+1;
trueclass=zeros(length(v_trueclass),max(v_trueclass));
for i=1:length(v_trueclass)
    trueclass(i,1:v_trueclass(i))=[dados.inicio(i):dados.fim(i)];
end
clear EAC;clear CE;
filename=[path_fich fich];
load(filename,'-mat'); %load da matriz de coassocs

co_assocs= exp_1.CE.co_assocs + exp_1.CE.co_assocs' + speye(ns,ns).*exp_1.CE.co_assocs_norm_fact;
co_assocs=co_assocs/exp_1.CE.co_assocs_norm_fact;
co_assocs_f=full(co_assocs); %full para metodo classico
methods={'single'};methods_acronym={'SL'}
for o=1:length(methods)
    method=methods{o};
    Z=apply_hierq2nassocs(co_assocs_f,[],method);
    nsamples_in_cluster=[];clusters_m=[];
    %k-fixo
    [nsamples_in_cluster,clusters_m]= get_nc_clusters_from_SL_dendro(Z,dados.nc,[]);
    hit_counter=determine_ci(trueclass,clusters_m,ns);
    EAC.(methods_acronym{o}).comb_partition_fixedk.nsamples_in_cluster=nsamples_in_cluster
    EAC.(methods_acronym{o}).comb_partition_fixedk.clusters_m=clusters_m;
    EAC.(methods_acronym{o}).comb_partition_fixedk.nc=dados.nc;
    EAC.(methods_acronym{o}).comb_partition_fixedk.ci=hit_counter;
    resumo_n(n,o)=hit_counter;
end
%fazer conv para dissemelhanca antes de converter para esparsa
delta=eps;
[IsMax]=find(co_assocs_f==1);
co_assocs_d=1-co_assocs_f;
co_assocs_d(IsMax)=delta;
[i,j,s]=find(co_assocs_d);
%s=1-s;
co_assocs=sparse(i,j,s,ns,ns);
figure;imagesc(co_assocs)
acr='MST';
T = mst(co_assocs);
figure;imagesc(T);colorbar
%[out,clusters]=extract_K_MST2(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
[out,clusters]=extract_K_MST3(T,dados.nc,ns); %estou a usar o n_c: numero de clusters reais...
for k=1:max(clusters)
    I=find(clusters==k);
    EAC.(acr).comb_partition_fixedk.nsamples_in_cluster(k)=length(I);
    EAC.(acr).comb_partition_fixedk.clusters_m(k,1:length(I))=I;
end
determine_ci(trueclass,EAC.(acr).comb_partition_fixedk.clusters_m,ns)

EAC.SL.comb_partition_fixedk.nsamples_in_cluster
EAC.MST.comb_partition_fixedk.nsamples_in_cluster
