import pandas as pd
import numpy as np
import sys
import pickle
import warnings
import xlrd
import json
from collections import Counter
from tqdm import tqdm_notebook as tqdmn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import helpers_classifiers as help_class
import helpers_text_semantics as help_txt

if __name__ == '__main__':
    print("Parsing Arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',help = 'Output filename',default="")
    args = parser.parse_args()
    #Data Generation
    #
    base_dir = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/"
    #
    ## Semantics & Syntax
    d100 = pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
    usr_tweet_text = help_txt.generate_tw_semantic_info(base_dir + "data_files/UKSOC_rep/tweets/all_geolocated_users.csv",d100)
    usr_profile_data = help_txt.generate_profile_information()
    df_usr_profile_tweets = pd.merge(usr_profile_data,usr_tweet_text,left_on="id",right_on="user_id")
    df_usr_profile_tweets = help_txt.generate_full_features(df_usr_profile_tweets,min_tweets=50)
    #
    ## SES enrichment
    usr2ses = pd.read_csv(base_dir + ""data_files/UKSOC_rep/archi_ses_data.csv")
    ses_text_archi = pd.merge(df_usr_profile_tweets,usr2ses,left_on="id",right_on="user_id")
    ses_texses_text_archit_occ.dropna(subset=["bianswer"],inplace=True)
    ses_insee_class = np.array(ses_text_archi.bianswer).astype(np.int)# 2 class
    #
    # Model Fitting
    mat_info = np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_insee.iloc[0]["fts"])))
                        for it,sample in (ses_text_insee[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    dic_res=help_class.test_all_models(X, ses_insee_class)
    pickle.dump(dic_res, open( "/warehouse/COMPLEXNET/jlevyabi/tmp/test_archi_%s.p"%(args.output), "wb" ))
