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

f __name__ == '__main__':
    #Data Generation
    #
    ## Location + SES
    base_dir = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/"
    d100 = pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
    usr_tweet_text = help_txt.generate_tw_semantic_info(base_dir+"data_files/UKSOC_rep/linkedin/linkedin_data/all_linkedin_users.csv",d100)
    usr_profile_data = help_txt.generate_profile_information()
    df_usr_profile_tweets = pd.merge(usr_profile_data,usr_tweet_text,left_on="id",right_on="user_id")
    df_usr_profile_tweets = help_txt.generate_full_features(df_usr_profile_tweets)

    ses_text_insee=pd.merge(df_usr_profile_tweets,usr2ses,left_on="id",right_on="usr")
    ses_text_insee.dropna(subset=["insee_iris_med"],inplace=True)
    ses_insee_class=np.array(ses_text_insee.insee_income> np.nanmean(ses_text_insee.insee_income)).astype(np.int)# 2 class
    mat_info=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_insee.iloc[0]["fts"])))
                        for it,sample in (ses_text_insee[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    print(help_class.generate_full_features(X, ses_insee_class))
