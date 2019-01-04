import pandas as pd
import numpy as np
import sys
import pickle
import warnings
import xlrd
import json
from collections import Counter

from tqdm import tqdm_notebook as tqdmn
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd

import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report

import helpers_locs_to_home as help_loc

france=Polygon([[-4.9658203125,42.3585439175],[8.4375,42.3585439175],[8.4375,51.2344073516],[-4.9658203125,51.2344073516],[-4.9658203125,42.3585439175]])
french_stopwords = list(set(stopwords.words('french')))
eng_stopwords = list(set(stopwords.words('english')))

""" Extract topic proportions per users given a clustered word embedding"""
def get_cluster_info(dic_clus,df_tweets):
    nb_clusters=len(list(dic_clus.keys()))
    word2cluster_only_pos={word:cluster_nb for cluster_nb,cluster_words in dic_clus.items() for word in cluster_words}
    clust_freq_only_pos=[]
    for tweet in tqdm(df_tweets.tweet_text):
        clust_freq_only_pos.append((Counter([word2cluster_only_pos[word]
                                             for word in tweet.split() if word in word2cluster_only_pos])))
    cfd_only_pos=[{k:(v+0.0)/(sum(dic_count.values()))for k,v in dic_count.items()}
                  for dic_count in clust_freq_only_pos]
    df_tweets["cfd_%d"%nb_clusters]=[np.array(list({clus:(dic_count[clus] if clus in dic_count else 0)
                                    for clus in range(len(dic_clus))}.values())) for dic_count in cfd_only_pos]
    return df_tweets

""" Extract excel data while ignoring first data rows"""
def extract_xls_file(path,sheet_name,offset):
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)
    rows = []
    for i, row in (enumerate(range(worksheet.nrows))):
        if i <= offset:  # (Optionally) skip headers
            continue
        r = []
        for j, col in enumerate(range(worksheet.ncols)):
            r.append(worksheet.cell_value(i, j))
        rows.append(r)
    return(pd.DataFrame(rows[1:],columns=rows[0]))

""" Yields back GeoDataFrame of geolocated tweets posted within France """
def geotreatment_french_tweets(data_fname):
    df_geotweets=pd.read_csv(data_fname,sep="\t",header=-1,index_col=False)
    df_geotweets.columns = ["id","time","lat","lon","geo_pt","service","profile","follows","friends","nb urls","loc_name","geo_type"]
    fechas,days,hours,minutes,seconds,years,months=help_loc.time_2_date(df_geotweets.time)
    df_geotweets['day']=days
    df_geotweets['hour']=hours
    df_geotweets['min']=minutes
    df_geotweets['sec']=seconds
    df_geotweets['year']=years
    df_geotweets['month']=months
    df_geotweets['fecha']=fechas
    df_geotweets["geometry"]=[Point((float(lon),float(lat))) for lon,lat in zip(df_geotweets.lon,df_geotweets.lat)]
    df_geotweets=gpd.GeoDataFrame(df_geotweets,crs = {'init': 'epsg:4326'})
    df_geotweets_france=df_geotweets[[france.contains(geo_pt) for geo_pt in df_geotweets.geometry]]
    print("Number of geolocated tweets during 2014-2015... %d geolocations"% df_geotweets.shape[0])
    print("Number of geolocated tweets during 2014-2015 in France... %d geolocations"% df_geotweets_france.shape[0])
    return df_geotweets_france

""" Computes income per IRIS block """
def generate_iris_ses_data(f_base="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/geoloc/iris_opendata/"):
    f1 = f_base + "BASE_TD_FILO_DISP_IRIS_2013.xls"
    d_filo_disp_iris = extract_xls_file(f1,"IRIS_DISP",4)
    f2 = f_base + "BASE_TD_FILO_DEC_IRIS_2013.xls"
    d_filo_dec_iris = extract_xls_file(f2,"IRIS_DEC",4)
    geo_file = f_base + "contours-iris-2016.geojson"
    df_geo_iris = gpd.read_file(geo_file)
    d_iris=df_geo_iris[[france.contains(geo_pt) if geo_pt else False for geo_pt in (df_geo_iris.geometry)]]
    d_iris['IRIS']=d_iris.code_iris
    dec_income_iris=pd.merge(d_iris, d_filo_dec_iris, on='IRIS')
    return dec_income_iris

""" Filters out non-reliable users and computes home location"""
def reliable_home_location(usrs_with_SES_info_dic,max_km_var=10,max_km_per_h=120,nb_mini_locs=5,nb_min_crazy=20,thresh_rate=3):
    dic_locs_reals, _, _, _, _, _, _ = help_loc.fast_get_repr_location(dic_locs=usrs_with_SES_info_dic,max_km_var=max_km_var,
                                                                       max_km_per_h=max_km_per_h,nb_mini_locs=nb_mini_locs,nb_min_crazy=nb_min_crazy)
    new_dic_real, _ = help_loc.remove_hyperactive_usrs(dic_locs_reals,pandas_version=0,thresh_rate=thresh_rate)
    new_dic_real, _ = help_loc.remove_hyper_social_usrs(dic_real=new_dic_real)
    dic_pd = {k:pd.DataFrame(v,columns=["lat","lon","day","hour","minu","sec","year","month","fecha"]) for k,v in tqdmn(new_dic_real.items())}
    home_most_freq_all = help_loc.go_through_home_candidates(new_dic_real,help_loc.take_most_frequent_thresh)
    home_most_freq_night = help_loc.go_through_home_candidates(new_dic_real,help_loc.take_most_frequent_night_thresh)
    dic_all_users_insee={usr:{"profile":(new_dic_real[usr].profile),
                              "locations":new_dic_real[usr][["lat","lon","text","day", "hour","minu","sec", "year","month","fecha", "geo_pt","service"]],
                              "inferred_loc":home_most_freq_all[usr][["lat","lon"]],
                              "suppl_info":home_most_freq_all[usr]}
                         for usr in tqdm(home_most_freq_all.keys())}
    return dic_all_users_insee

""" Reads profile information and vectorizes using tf-idf"""
def generate_profile_information(fbase = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/UKSOC_rep/",max_fts=450):
    #Bio text data
    dic_bio=pickle.load(open(fbase+"all_together_profiles.p","rb"))
    #N-Gram vectorizer
    n_grams_bio_vect=TfidfVectorizer(stop_words=french_stopwords+eng_stopwords,max_features=max_fts,ngram_range=(1,2),lowercase=True)
    #Clean profile info
    tweet_clean = lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",unidecode.unidecode(x.replace("_",""))).split())
    #Generate profile info dataframe
    cols_of_interest=["id","followers_count","friends_count","listed_count","favourites_count","statuses_count","description"]
    profile_data=[[usr[0]._json[k] for k in cols_of_interest] for usr in dic_bio ]
    df_profile_data=pd.DataFrame(profile_data,columns=cols_of_interest)
    df_profile_data["description"]=df_profile_data.description.apply(tweet_clean)
    n_grams_bio=n_grams_bio_vect.fit_transform(list(df_profile_data.description.values))
    return n_grams_bio

""" Synthetize sematics features for tweets (topics + count-vectorizer)"""
def generate_tw_semantic_info(geolocated_tweets,word2topics_dic,max_fts=560):
    usr_tweet_text=pd.read_csv(geolocated_tweets,sep=';',header=0,)
    usr_text=(usr_tweet_text.dropna(how="any").drop(
        ["tweet_id","tweet_date"],axis=1).groupby('user_id',squeeze=True,)['tweet_text'].apply(lambda x: "%s" % ' '.join(x))).to_frame()
    usr_text.reset_index(inplace=True,drop=False)
    n_grams_tweet_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,max_features=max_fts,ngram_range=(1,1),lowercase=True)
    n_grams_tweet=n_grams_tweet_vect.fit_transform(list(usr_text.tweet_text.values))
    usr_text=get_cluster_info(word2topics_dic,usr_text);
    return usr_text, n_grams_tweet

if __name__ == '__main__':
    print()
