echo "Generating word to topic association"
w2v_model="/home/jlevyabi/seacabo/data_files/lowe_dim_sosweet2vec.w2v"
spec_clust_out="/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/spec_corrected_clusters_only_pos_entries.p"
~/anaconda3/bin/python ../python_scripts/spec_clust_w2v_zero_negs.py -w2v  $w2v_model -out $spec_clust_out -nbc 100





