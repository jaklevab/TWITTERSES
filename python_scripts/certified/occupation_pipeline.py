import pandas as pd
import numpy as np
import sys
import pickle
import warnings
import xlrd
import json
from collections import Counter
import argparse

from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import helpers_locs_to_home as help_loc
import helpers_classifiers as help_class
import helpers_text_semantics as help_txt
import helpers_ses_enrichment as help_ses

if __name__ == '__main__':
    print("Parsing Arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',help = 'Output filename',default="")
    parser.add_argument('-njbs', '--njbs',help = 'Number of jobs to parallelize over',default=-1)
    args = parser.parse_args()
    n_jobs = int(args.njbs)
    #Data Generation
    #
    base_dir = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/"
    #
    ## Semantics & Syntax
    d100 = pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
    usr_tweet_text = help_txt.generate_tw_semantic_info(base_dir+"data_files/UKSOC_rep/linkedin/linkedin_data/all_linkedin_users.csv",d100)
    usr_profile_data = help_txt.generate_profile_information()
    df_usr_profile_tweets = pd.merge(usr_profile_data,usr_tweet_text,left_on="id",right_on="user_id")
    df_usr_profile_tweets = help_txt.generate_full_features(df_usr_profile_tweets,min_tweets=50)
    #
    ## SES enrichment
    usr2ses = pd.read_csv(base_dir + "icdm18/issues/linkedin_salaries.csv")
    ses_text_occ = pd.merge(df_usr_profile_tweets,usr2ses,left_on="id",right_on="user_id")
    ses_text_occ.dropna(subset=["estimated_sal"],inplace=True)
    ses_occ_class = np.array(ses_text_occ.estimated_sal> np.nanmedian(ses_text_occ.estimated_sal)).astype(np.int)# 2 class
    #
    # Model Fitting
    mat_info = np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_occ.iloc[0]["fts"])))
                        for it,sample in (ses_text_occ[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    dic_res=help_class.test_all_models(X, ses_occ_class,n_jobs=n_jobs)
        pickle.dump(dic_res, open( "/warehouse/COMPLEXNET/jlevyabi/tmp/test_occupation_%s.p"%(args.output), "wb" ))
