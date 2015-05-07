
This is a set of MATLAB m-files implementing Evidence accumulation clustering algorithm described in the paper A. Fred and A.K.Jain, "Data clustering using evidence accumulation", ICPR2002.

It consists of a main MATLAB function called "combina_generico2a.m" and
several auxiliary functions: "update_assoc_mats3b.m", "apply_hierq2nassocs1.m", "get_nc_clusters_from_SL_dendro.m", and "get_nc_stable_from_SL_dendro.m", which are called by the main program. The clustering ensemble partitions are produced using "k_medias_with_seed_vns.m".

For instructions type "help combina_generico2a" at the MATLAB prompt,
or read the first few lines of the "combina_generico2a.m" file.

Also included are one simple demo (demo.m) which exemplify how to use the
program. 