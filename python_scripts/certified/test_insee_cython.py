import re
import pandas as pd
import numpy as np
import sys
import geopandas as gpd
from geopandas import GeoDataFrame
from geopandas import sjoin
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
from fast_pt_in_poly import contains_cy_insee

# Declare INSEE data
print("INSEE DECLARATION ...")
pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''')
new_geo=[]
df_insee=pd.read_csv('/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/geoloc/final_exact.csv',sep=";")
#Cast data into shapely format
for geo in tqdm(df_insee.geometry):
    matches = pat.findall(geo)
    if matches:
        lst = Polygon([tuple(map(float, m.split())) for m in matches])
    else:
        lst=None
    new_geo.append(lst)

presentable_insee_df=df_insee
presentable_insee_df=presentable_insee_df.convert_objects(convert_numeric=True)
presentable_insee_df['geometry']=new_geo
geo_insee = GeoDataFrame(presentable_insee_df)
geo_insee.crs={'init': 'epsg:4326'}

lat,lon = [],[]
for poly in tqdm(geo_insee.geometry ):
    for coor_pair in poly.exterior.coords[:]:
        min
        lat.append(coor_pair[1])
        lon.append(coor_pair[0])
min_lat,max_lat=np.min(lat),np.max(lat)
min_lon,max_lon=np.min(lon),np.max(lon)
del lon;del lat

prec=2
map_prec = lambda x: str(round(x,prec))
step=10**(-prec)
lat_grid = np.arange(start=min_lat, stop=max_lat+step,step= step)
lon_grid = np.arange(start=min_lon,stop=max_lon+step,step=step)

geo_insee_dic_KEYS=[]
for my_lat in (lat_grid):
    for my_lon in lon_grid:
        lat2str,lon2str=map_prec(my_lat),map_prec(my_lon)
        geo_insee_dic_KEYS.append((lat2str,lon2str))

geo_insee_dic={key:[] for key in geo_insee_dic_KEYS}
for it,poly in (geo_insee.iterrows()):
    center_x,center_y=poly.geometry.centroid.bounds[:2]
    cx,cy=map_prec(center_x),map_prec(center_y)
    if cx=="-0.0":
        cx="0.0"
    geo_insee_dic[(cy,cx)].append(poly)

# Divide original GeoDF into small geodfs for each patch of territory
for k,v in (geo_insee_dic.items()):
    df_coord=GeoDataFrame(v);df_coord.crs={'init': 'epsg:4326'}
    geo_insee_dic[k]=df_coord



def my_pt2poly(usr_df,country_dic,set_keys):
    usr_info=[];prec=2;step=10**(-prec);vals=[-step,0,step]
    test=[]
    map_prec = lambda x: str(round(x,prec))
    for it,usr in tqdm(usr_df.iterrows()):
        us_posx,us_posy=usr.geometry.centroid.bounds[:2]
        usr_geom=usr.geometry._geom
        keys=(set([(map_prec(us_posy+yval),map_prec(us_posx+xval)) for xval in vals for yval in vals])).intersection(set_keys)
        pre_df_of_concern=[country_dic[key] for key in keys if country_dic[key].shape[0]>0]
        df_of_concern=[y._geom for x in pre_df_of_concern for y in x.geometry]
        df_ilocs_concern=[y for x in pre_df_of_concern for y in x["Unnamed: 0"]]
        if len(df_of_concern)==0:
            test.append(None)
            continue
        geom_to_check = df_of_concern
        assignments=(contains_cy_insee(np.array([usr_geom]), np.array(geom_to_check)))
        to_check=np.where(assignments)
        _,poly=to_check
        if len(poly)==0:
            test.append(None)
        else:
            test.append(df_ilocs_concern[poly[0]])
    return test

print("TWITTER DATA ...")
data_prof_14=pd.read_csv(
    "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/icdm18/issues/icdm_geousers_profile_14.txt",
    sep="\t",header=-1,names=["id","time","lat","lon","geo_pt","service","profile","follows","friends","nb urls","loc_name","geo_type"],
    index_col=False)


##### Users in France
france=Polygon([[-4.9658203125,42.3585439175],[8.4375,42.3585439175],
                [8.4375,51.2344073516],[-4.9658203125,51.2344073516],
                [-4.9658203125,42.3585439175]])

data_prof_14["geometry"]=[Point(x.lon,x.lat) for it,x in tqdm(data_prof_14[["lon","lat"]].iterrows())]
dgeo_prof_france_14=data_prof_14[[france.contains(geo_pt) for geo_pt in data_prof_14.geometry]]
print("CYTHON TEST ...")
loc2insee=my_pt2poly(dgeo_prof_france_14,geo_insee_dic,set(geo_insee_dic_KEYS))
