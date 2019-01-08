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

france=Polygon([[-4.9658203125,42.3585439175],[8.4375,42.3585439175],[8.4375,51.2344073516],[-4.9658203125,51.2344073516],[-4.9658203125,42.3585439175]])

""" Yields back GeoDataFrame of geolocated tweets posted within France """
def geotreatment_french_tweets(data_fname):
    df_geotweets=pd.read_csv(data_fname,sep="\t",header=-1,index_col=False)
    df_geotweets.columns = ["id","time","lat","lon","geo_pt","service","profile","follows","friends","tweet","nb urls"]
    fechas,days,hours,minutes,seconds,years,months=help_loc.time_2_date(df_geotweets.time)
    df_geotweets['day']=days
    df_geotweets['hour']=hours
    df_geotweets['minu']=minutes
    df_geotweets['sec']=seconds
    df_geotweets['year']=years
    df_geotweets['month']=months
    df_geotweets['fecha']=fechas
    df_geotweets["geometry"]=[Point((float(lon),float(lat))) for lon,lat in tqdm(zip(df_geotweets.lon,df_geotweets.lat))]
    df_geotweets=gpd.GeoDataFrame(df_geotweets,crs = {'init': 'epsg:4326'})
    df_geotweets_france=df_geotweets[[france.contains(geo_pt) for geo_pt in tqdm(df_geotweets.geometry)]]
    print("Number of geolocated tweets during 2014-2015... %d geolocations"% df_geotweets.shape[0])
    print("Number of geolocated tweets during 2014-2015 in France... %d geolocations"% df_geotweets_france.shape[0])
    return df_geotweets_france

""" Takes disordered df and returns dic factorized by user id"""
def factorize_income_data(usrs_with_income):
    usrs_with_SES_info_dic={}
    for it,row in tqdm(usrs_with_income.iterrows()):
        _=usrs_with_SES_info_dic.setdefault(row.id,[]);
        usrs_with_SES_info_dic[row.id].append(row)
    for usr,val in tqdm(usrs_with_SES_info_dic.items()):
        usrs_with_SES_info_dic[usr]=pd.DataFrame(val,columns=list(usrs_with_income.columns))
    return usrs_with_SES_info_dic

if __name__ == '__main__':
    print("Parsing Arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-ses', '--ses_data',help = 'Source for SES Map (iris or insee)',
                  default="iris",choices=['insee','iris'])
    parser.add_argument('-o', '--output',help = 'Output filename',default="")
    args = parser.parse_args()
    ses_source = args.ses_data
    #Data Generation
    #
    ## Location + SES
    base_dir = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/"
    data_geo_france = geotreatment_french_tweets(base_dir + "icdm18/issues/icdm_geousers_profile_14.txt")
    dec_income = help_ses.generate_iris_ses_data() if ses_source == 'iris' else help_ses.generate_insee_ses_data()
    #
    ## Semantics & Syntax
    d100 = pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
    usr_tweet_text = help_txt.generate_tw_semantic_info(base_dir + "data_files/UKSOC_rep/tweets/all_geolocated_users.csv",d100)
    usr_profile_data = help_txt.generate_profile_information()
    df_usr_profile_tweets = pd.merge(usr_profile_data,usr_tweet_text,left_on="id",right_on="user_id")
    df_usr_profile_tweets = help_txt.generate_full_features(df_usr_profile_tweets,min_tweets=50)
    #
    ## Location Filtering + SES enrichment
    usrs_with_income = gpd.sjoin(data_geo_france, dec_income,op='within') if ses_source == 'iris' else help_ses.insee_sjoin(dgeo_prof_france_14,dec_income)
    usrs_with_SES_info_dic = factorize_income_data(usrs_with_income)
    income_str = "DEC_MED13" if ses_source == 'iris' else "income"
    usr2ses = help_ses.reliable_home_location(usrs_with_SES_info_dic,income_str)
    ses_text_insee=pd.merge(df_usr_profile_tweets,usr2ses,left_on="id",right_on="usr")
    ses_text_insee.dropna(subset=["insee_iris_inc"],inplace=True)
    ses_insee_class=np.array(ses_text_insee.insee_iris_inc> np.nanmedian(ses_text_insee.insee_iris_inc)).astype(np.int)# 2 class
    #
    # Model Fitting
    mat_info=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_insee.iloc[0]["fts"])))
                        for it,sample in (ses_text_insee[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    dic_res=help_class.test_all_models(X, ses_insee_class)
    pickle.dump(dic_res, open( "/warehouse/COMPLEXNET/jlevyabi/tmp/test_location_%s.p"%(args.output), "wb" ))
