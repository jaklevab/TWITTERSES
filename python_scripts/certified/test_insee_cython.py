import time
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

import helpers_locs_to_home as help_loc
import helpers_classifiers as help_class
import helpers_text_semantics as help_txt
import helpers_ses_enrichment as help_ses


geo_insee_dic = help_ses.generate_insee_ses_data()

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
start_time = time.time()
loc2insee=help_ses.insee_sjoin(dgeo_prof_france_14,geo_insee_dic)
elapsed_time = time.time() - start_time
start_time = time.time()
elapsed_time = time.time() - start_time

