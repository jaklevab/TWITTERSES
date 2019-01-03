################################# IMPORTS #################################

import pandas as pd
import numpy as np
from datetime import date,timedelta
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from scipy.spatial import cKDTree
from collections import Counter
from geopy.distance import vincenty
from datetime import datetime
import time
from pyproj import Proj, transform
from scipy.spatial import cKDTree
from scipy import sparse
import sqlite3 as lite
from collections import Counter
import json
from tqdm import tqdm_notebook as tqdm
import sys
import os
#path = "/datastore/complexnet/twitter/data/users.db"
#con = lite.connect(path)
import warnings
warnings.filterwarnings("ignore")

uk = '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 \
+x_0=400000 +y_0=-100000 +ellps=airy \
+towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 +units=m +no_defs'

################################# TODO #################################

# INCREASE nb_mini_locs !!! --> 15
# I've changed the parts where I read the sql db because of some user permission problem :Change
print("Reminder: Not running filters based on users.db")
################################# HELPERS #################################
"""Take string of data returns arrays for day, ..., year"""
def time_2_date(time_array):
    fechas=[];days=[];hours=[];minutes=[];seconds=[];years=[];months=[]
    for times in tqdm(time_array):
        try:
            tab_date=times.split('-')
            fecha=(date(int(tab_date[0]),int(tab_date[1]),int(tab_date[2][:2]))).weekday()
            hms=(times.split('T')[1])
            inter=hms.split(':')
            hour=inter[0];minute=inter[1];sec=inter[2][:2]
            year=tab_date[0]
            month=tab_date[1]
            day=tab_date[2][:2]
            years.append(year)
            days.append(day)
            fechas.append(fecha)
            hours.append(int(hour));minutes.append(int(minute));seconds.append(int(sec[:1]))
            months.append(month)
        except:
            print(times)
    return days,fechas,hours,minutes,seconds,years,months

"""Project geographic co-ordinates to get cartesian x,y. Transform(origin|destination|lon|lat)"""
def proj_arr(points,proj_to):
    inproj = Proj(init='epsg:4326')
    outproj = Proj(proj_to)
    func = lambda x: transform(inproj,outproj,x[1],x[0])
    return np.array(list(map(func, points)))

""" Time difference to hours """
def conv_to_hours(time_delt):
    return abs(time_delt.days)*24.0+time_delt.seconds/3600

""" Returns category of user (real, to fast, ...) """
def is_real(visit_df,max_km_var,max_km_per_h,nb_mini_locs):
    dists=[];N=visit_df.shape[0]
    if(N<=nb_mini_locs):
        return None,0,0,0
    points = np.array(visit_df)[:,:2].astype(float).tolist()
    proj_pnts = proj_arr(points, uk)
    tree = cKDTree(proj_pnts)
    cut_off_metres = max_km_var*(3e3) #90KM (our projection is in metres!)
    tree_dist = tree.sparse_distance_matrix(tree,cut_off_metres,p=2)
    spa_dists = 0.001*sparse.tril(tree_dist, k=-1).todense()   # zero the main diagonal (distance=0)
    times = [datetime(val[4],val[5],val[6],val[1],val[2],val[3]) for val in np.array(visit_df)[:,2:].astype(int).tolist()]
    temp_dist=np.array([conv_to_hours(t1-t2) for t1 in times for t2 in times]).reshape((N,N))
    speed=spa_dists/temp_dist;speed=speed[np.isfinite(speed)]
    if len(speed)>1 and np.nanmax(speed)>max_km_per_h:
        return False,0,med,speed
    rel_dist=spa_dists[np.tril_indices(N,k=-1)]
    med=np.mean(rel_dist)
    if med>max_km_var:
        return False,1,med,speed
    return True,"",rel_dist,speed

""" Signals too fast/variable/maree_info accounts  """
def filter_crazy_users(dic,max_km_var,max_km_per_h,nb_mini_locs,nb_min_crazy):
    dic_too_fast={}
    dic_too_var={}
    dic_mar={}
    dic_dist={};dic_speed={}
    dic_real_usrs={}
    with pd.option_context('display.max_rows', None, 'display.max_columns', 30):#with con:
        for usr,visits in tqdm(dic.items()):
            #cur = con.cursor()
            #cur.execute("SELECT screen_name FROM users WHERE id = '%s'" %usr )
            s_name=None#cur.fetchone()
            if (s_name!=None and "maree_info" in s_name[0]):
                dic_mar[usr]=visits
                continue
            visits_info=pd.DataFrame(data=visits,columns=["lat","lon","day","hour","minu","sec","year","month","fecha"])
            if len(visits_info.lat)>nb_min_crazy and len(Counter([tuple(x)
                                                         for x in visits_info[["lat","lon"]].values.tolist()]))==1:
                dic_mar[usr]=visits
            ticket,code,dists,speed=is_real(visits_info,max_km_var,max_km_per_h,nb_mini_locs)
            if ticket==None:
                continue
            elif ticket:
                dic_dist[usr]=[np.mean(dists),np.max(dists),visits_info.shape[0]]
                dic_speed[usr]=[np.mean(speed)]
                dic_real_usrs[usr]=visits
            elif code==0:
                dic_dist[usr]=[np.mean(dists),np.max(dists),visits_info.shape[0]]
                dic_speed[usr]=[np.mean(speed)]
                dic_too_fast[usr]=visits
            else:
                dic_dist[usr]=[np.mean(dists),np.max(dists),visits_info.shape[0]]
                dic_too_var[usr]=visits
    return dic_real_usrs,dic_too_fast,dic_too_var,dic_mar,dic_dist,dic_speed

"""Pandas to dictionary of geolocation organized by user"""
def fact_geo_frame_by_usr(d_home):
    dic_locs={}
    for index,row in tqdm(d_home.iterrows()):
        usr,lat,lon,day,hour,mi,sec,year,month,fecha=(row.usr,row.lat,row.lon,
                                               row.day,row.hour,row["min"],row.sec,row.year,row.month,row.fecha)
        if usr not in dic_locs:dic_locs[usr]=[(lat,lon,day,hour,mi,sec,year,month,fecha)]
        else:dic_locs[usr].append((lat,lon,day,hour,mi,sec,year,month,fecha))
    return dic_locs

""" Removes unreal users and returns user with most frequent location"""
def fast_get_repr_location(dic_locs,max_km_var=10,max_km_per_h=900,nb_mini_locs=3,nb_min_crazy=20):
    d_real,d_fast,d_vars,dic_mar,dic_dist,dic_speed=filter_crazy_users(dic_locs,max_km_var,max_km_per_h,nb_mini_locs,nb_min_crazy)
    res=[]
    for usr,visits in tqdm(d_real.items()):
        visit_usrs=Counter(visits)
        most_freq=visit_usrs.most_common(1)[0]
        res.append((usr,most_freq[0][0],most_freq[0][1],most_freq[1],sum(visit_usrs.values()),
                    (most_freq[1]+0.0)/sum(visit_usrs.values())))
    return(d_real,d_fast,d_vars,dic_mar,dic_dist,dic_speed,pd.DataFrame(data=res,columns=["usr","lat","lon","nb_loc","total_geo","proba"]))

""" Time difference to seconds """
def conv_to_hours_precise(date_1,date_2):
    y_1,mo_1,d_1,h_1,mi_1,s_1=date_1;d1=date(int(y_1),int(mo_1),int(d_1))
    y_2,mo_2,d_2,h_2,mi_2,s_2=date_2;d2=date(int(y_2),int(mo_2),int(d_2))
    return abs((d1-d2).seconds)+abs(int(h_1)-int(h_2))*3600+abs(int(mi_1)-int(mi_2))*60+abs(int(s_1)-int(s_2))

""" Removes users which 2nd most frequent location appears more than 40% times   """
def remove_active_static_usrs(dic_real,pandas_version=1,min_locs=10,thresh=0.4):
    suspected_automated_usrs=[]
    if pandas_version==1:
        for usr, locs in (dic_real.items()):
            geos_tuples = Counter([tuple(x) for x in (locs[["lat","lon"]]).values])
            if (locs.shape[0]>=min_locs) and (geos_tuples.most_common(1)[1])>=int(thresh*locs.shape[0]):
                suspected_automated_usrs.append(usr)
    else:
        for usr, locs in (dic_real.items()):
            geos_tuples = Counter([(x[0],x[1]) for x in locs])
            if (len(locs)>=min_locs) and (geos_tuples.most_common(1)[0][1])>=int(thresh*len(locs)):
                suspected_automated_usrs.append(usr)
    new_dic_real={k:v for k,v in dic_real.items() if k not in suspected_automated_usrs}
    return new_dic_real,suspected_automated_usrs

""" Removes users posting 2 consecutive tweets in less than 2 secs   """
def remove_hyperactive_usrs(dic_real,pandas_version=1,thresh_rate=2):
    suspected_automated_usrs=[]
    if pandas_version==1:
        for usr, locs in (dic_real.items()):
            time_activity_curr=(locs[["year","month","fecha","hour","minu","sec"]]).values
            time_activity_next=(time_activity_curr.shift(1)).values
            for curr,foll in zip(time_activity_curr,time_activity_next):
                if conv_to_hours_precise(curr,foll)<=thresh_rate:
                    suspected_automated_usrs.append(usr)
                    break
    else:
        for usr, locs in (dic_real.items()):
            t_curr=pd.DataFrame(locs,columns=["lat","lon","day",
                                              "hour","minu","sec","year","month","fecha"])
            time_activity_curr=t_curr[["year","month","fecha","hour","minu","sec"]].values
            t_next=t_curr.shift(1);it=0
            time_activity_next=t_next[["year","month","fecha","hour","minu","sec"]].values
            for curr,foll in zip(time_activity_curr,time_activity_next):
                if it==0:
                    it+=1;continue
                if conv_to_hours_precise(curr,foll)<=thresh_rate:
                    suspected_automated_usrs.append(usr)
                    break
                it+=1
    new_dic_real={k:v for k,v in dic_real.items() if k not in suspected_automated_usrs}
    return new_dic_real,suspected_automated_usrs

""" Removes users whith more than 5k friends/folls  """
def remove_hyper_social_usrs(dic_real,friend_foll_thresh=5000):
    suspected_automated_usrs=[]
    with pd.option_context('display.max_rows', None, 'display.max_columns', 30):#with con:
        for usr in (dic_real.keys()):
            #cur = con.cursor()
            #cur.execute("SELECT friends FROM users WHERE id = '%s'" %usr )
            s_friends=[]#json.loads(cur.fetchone()[0])
            if (len(s_friends)>=friend_foll_thresh):
                suspected_automated_usrs.append(usr)
    new_dic_real={k:v for k,v in dic_real.items() if k not in suspected_automated_usrs}
    return new_dic_real,suspected_automated_usrs

""" User to SES of most frequent location  """
def moded_location_to_income(dic_usr_locs,dic_soc_info,precision):
    new_dic_usr_loc={}
    for k,v in tqdm(dic_usr_locs.items()):
        usr_info=np.array(v)[:,:2].astype(float).tolist()
        locs=[((round(lat,precision)),(round(lon,precision)))
              for lat,lon in usr_info if (str(round(lat,precision)),str(round(lon,precision))) in dic_soc_info.keys()]
        if len(locs)==0:
            continue
        count_locs=Counter(locs)
        most_common=count_locs.most_common(1)[0]
        lat,lon=most_common[0]
        new_dic_usr_loc[k]=(most_common[0],dic_soc_info[(str(round(lat,precision)),str(round(lon,precision)))]['income'],
                            most_common[1]/sum(np.array(list(count_locs.values()))+0.0))
    return new_dic_usr_loc

""" User to SES of average location  """
def location_to_weighted_income(dic_usr_locs,dic_dist,dic_soc_info,precision):
    new_dic_usr_loc={}
    for k,v in tqdm(dic_usr_locs.items()):
        usr_info=np.array(v)[:,:2].astype(float).tolist()
        locs=[((round(lat,precision)),(round(lon,precision)))
              for lat,lon in usr_info if (str(round(lat,precision)),str(round(lon,precision))) in dic_soc_info.keys()]
        if len(locs)==0:
            continue
        count_locs=Counter(locs)
        most_common=count_locs.most_common(1)[0]
        most_common_count=most_common[1]
        incomes=[dic_soc_info[(str(round(lat,precision)),str(round(lon,precision)))]['income'] for lat,lon in locs ]
        insert=None
        if k in dic_dist.keys():
            insert=dic_dist[k]
        new_dic_usr_loc[k]=(np.mean(incomes),np.std(incomes),insert)
    return new_dic_usr_loc

