import pandas as pd
import numpy as np
import sys
import pickle
import warnings
import xlrd
import json
from collections import Counter
import re

from tqdm import tqdm
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import geopandas as gpd
from geopandas import sjoin
from fast_pt_in_poly import contains_cy_insee

import helpers_locs_to_home as help_loc

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

""" Computes income per INSEE block """
def generate_insee_ses_data(f_data="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/geoloc/final_exact.csv",prec=2):
    step=10**(-prec)
    map_prec = lambda x: str(round(x,prec))
    pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')
    new_geo=[]
    presentable_insee_df=pd.read_csv(f_data,sep=";")
    presentable_insee_df.rename({"Unnamed: 0":"position"},inplace=True)
    new_geo = [Polygon([tuple(map(float, m.split())) for m in pat.findall(geo)])
                       if pat.findall(geo) else None for geo in tqdm(presentable_insee_df.geometry)]
    presentable_insee_df=presentable_insee_df.convert_objects(convert_numeric=True)
    presentable_insee_df['geometry']=new_geo
    geo_insee = GeoDataFrame(presentable_insee_df,crs={'init': 'epsg:4326'})
    geo_insee_dic={}
    for it,row in geo_insee.iterrows():
        center_x,center_y=map(map_prec,row.geometry.centroid.bounds[:2])
        if center_x=="-0.0":
            center_x="0.0"
        _=geo_insee_dic.setdefault((center_y,center_x),[])
        geo_insee_dic[(center_y,center_x)].append(row)
    # Divide original GeoDF into small geodfs for each patch of territory
    return {k:GeoDataFrame(v,crs={'init': 'epsg:4326'},columns=["geometry"]) for k,v in geo_insee_dic.items()}

""" Optimized Cythonized Spatial Join for INSEE """
def insee_sjoin(usr_df,country_dic,prec=2):
    insee_corresp = [];step=10**(-prec);step=10**(-prec);vals = [-step,0,step]
    set_keys = set(country_dic.keys())
    map_prec = lambda x: str(round(x,prec))
    for it,usr in tqdm(usr_df.iterrows()):
        us_posx,us_posy=usr.geometry.centroid.bounds[:2]
        usr_geom = usr.geometry._geom
        keys=set([(map_prec(us_posy+yval),map_prec(us_posx+xval)) for xval in vals for yval in vals]).intersection(set_keys)
        pre_df_of_concern=[country_dic[key] for key in keys if country_dic[key].shape[0]>0]
        geom_to_check=[y._geom for x in pre_df_of_concern for y in x.geometry]
        df_ilocs_concern=[y for x in pre_df_of_concern for y in x.position]
        if len(geom_to_check)==0:
            insee_corresp.append(None)
            continue
        _,poly=np.where(contains_cy_insee(np.array([usr_geom]), np.array(geom_to_check)))
        if len(poly)==0:
            insee_corresp.append(None)
        else:
            insee_corresp.append(df_ilocs_concern[poly[0]])
    return insee_corresp

""" Filters out non-reliable users and computes home location"""
def reliable_home_location(usrs_with_SES_info_dic,income_str,max_km_var=10,max_km_per_h=120,nb_mini_locs=5,nb_min_crazy=20,thresh_rate=3):
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
    usr2ses=pd.DataFrame([[k,v["suppl_info"][income_str]] for k,v in dic_all_users_insee.items()],
                         columns=["usr","insee_iris_inc"]).dropna(how="any")
    return usr2ses
