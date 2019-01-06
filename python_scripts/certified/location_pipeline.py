import pandas as pd
import numpy as np
import sys
import pickle
import warnings
import xlrd
import json
from collections import Counter

from tqdm import tqdm
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import helpers_locs_to_home as help_loc
import helpers_classifiers as help_class
import helpers_text_semantics as help_txt

france=Polygon([[-4.9658203125,42.3585439175],[8.4375,42.3585439175],[8.4375,51.2344073516],[-4.9658203125,51.2344073516],[-4.9658203125,42.3585439175]])


""" Extract excel data while ignoring first data rows"""
def extract_xls_file(path,sheet_name,offset):
    workbook = xlrd.open_workbook(path)
    worksheet = workbook.sheet_by_name(sheet_name)
    rows = []
    for i, row in tqdm(enumerate(range(worksheet.nrows))):
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

""" Computes income per IRIS block """
def generate_iris_ses_data(f_base="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/geoloc/iris_opendata/"):
    f1 = f_base + "BASE_TD_FILO_DISP_IRIS_2013.xls"
    d_filo_disp_iris = extract_xls_file(f1,"IRIS_DISP",4)
    f2 = f_base + "BASE_TD_FILO_DEC_IRIS_2013.xls"
    d_filo_dec_iris = extract_xls_file(f2,"IRIS_DEC",4)
    geo_file = f_base + "shapefile_iris/contours-iris-2016.geojson"
    df_geo_iris = gpd.read_file(geo_file)
    d_iris=df_geo_iris[[france.contains(geo_pt) if geo_pt else False for geo_pt in tqdm(df_geo_iris.geometry)]]
    d_iris['IRIS']=d_iris.code_iris
    dec_income_iris=pd.merge(d_iris, d_filo_dec_iris, on='IRIS')
    return gpd.GeoDataFrame(dec_income_iris,crs = {'init': 'epsg:4326'})

""" Filters out non-reliable users and computes home location"""
def reliable_home_location(usrs_with_SES_info_dic,max_km_var=10,max_km_per_h=120,nb_mini_locs=5,nb_min_crazy=20,thresh_rate=3):
    dic_locs_reals, _, _, _, _, _, _ = help_loc.fast_get_repr_location(dic_locs=usrs_with_SES_info_dic,max_km_var=max_km_var,
                                                                       max_km_per_h=max_km_per_h,nb_mini_locs=nb_mini_locs,nb_min_crazy=nb_min_crazy)
    new_dic_real, _ = help_loc.remove_hyperactive_usrs(dic_locs_reals,pandas_version=0,thresh_rate=thresh_rate)
    new_dic_real, _ = help_loc.remove_hyper_social_usrs(dic_real=new_dic_real)
    dic_pd = {k:pd.DataFrame(v,columns=["lat","lon","day","hour","minu","sec","year","month","fecha"]) for k,v in tqdm(new_dic_real.items())}
    home_most_freq_all = help_loc.go_through_home_candidates(new_dic_real,help_loc.take_most_frequent_thresh)
    home_most_freq_night = help_loc.go_through_home_candidates(new_dic_real,help_loc.take_most_frequent_night_thresh)
    dic_all_users_insee={usr:{"profile":(new_dic_real[usr].profile),
                              "locations":new_dic_real[usr][["lat","lon","tweet","day", "hour","minu","sec", "year","month","fecha", "geo_pt","service"]],
                              "inferred_loc":home_most_freq_all[usr][["lat","lon"]],
                              "suppl_info":home_most_freq_all[usr]}
                         for usr in tqdm(home_most_freq_all.keys())}
    #
    usr2ses=pd.DataFrame([[k,v["suppl_info"]["DEC_D113"],
                       v["suppl_info"]["DEC_MED13"],v["suppl_info"]["DEC_D913"]]
                      for k,v in dic_all_users_insee.items()],
                      columns=["usr","insee_iris_lowe",
                               "insee_iris_med","insee_iris_sup"]).dropna(how="any")
    return usr2ses

if __name__ == '__main__':
    #Data Generation
    #
    ## Location + SES
    base_dir = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/"
    data_geo_france = geotreatment_french_tweets(base_dir + "icdm18/issues/icdm_geousers_profile_14.txt")
    dec_income_iris = generate_iris_ses_data()
    ## Semantics & Syntax
    d100 = pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
    usr_tweet_text = help_txt.generate_tw_semantic_info(base_dir + "data_files/UKSOC_rep/tweets/all_geolocated_users.csv",d100)
    usr_profile_data = help_txt.generate_profile_information()
    df_usr_profile_tweets = pd.merge(usr_profile_data,usr_tweet_text,left_on="id",right_on="user_id")
    df_usr_profile_tweets = help_txt.generate_full_features(df_usr_profile_tweets,min_tweets=50)
    #
    # Location Filtering + SES enrichment
    usrs_with_iris_income = gpd.sjoin(data_geo_france, dec_income_iris,how="inner", op='within')
    usrs_with_SES_info_dic={}
    for it,row in tqdm(usrs_with_iris_income.iterrows()):
        _=usrs_with_SES_info_dic.setdefault(row.id,[]);
        usrs_with_SES_info_dic[row.id].append(row)
    for usr,val in tqdm(usrs_with_SES_info_dic.items()):
        usrs_with_SES_info_dic[usr]=pd.DataFrame(val,columns=list(usrs_with_iris_income.columns))
    usr2ses = reliable_home_location(usrs_with_SES_info_dic)
    ses_text_insee=pd.merge(df_usr_profile_tweets,usr2ses,left_on="id",right_on="usr")
    ses_text_insee.dropna(subset=["insee_iris_med"],inplace=True)
    ses_insee_class=np.array(ses_text_insee.insee_iris_med> np.nanmean(ses_text_insee.insee_iris_med)).astype(np.int)# 2 class
    mat_info=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_insee.iloc[0]["fts"])))
                        for it,sample in (ses_text_insee[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    dic_res=help_class.test_all_models(X, ses_insee_class)
