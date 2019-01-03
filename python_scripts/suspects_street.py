# coding:utf-8
from __future__ import print_function
from __future__ import division
import requests
from tqdm import tqdm 
import pandas as pd
import time
import json
import sys,os

remove=input("Press 'y 'to remove previous versions of here created files.  ")
if remove =="y":
    os.system("rm -r /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/suspects_street/")
    os.system("rm /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/street_info.csv")
    os.system("mkdir /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/suspects_street/")

waiting=int(input("Write hours to wait before starting collection "))
time.sleep(3600*waiting)

# URLs antiguas
url_metadata="https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&size=800x600&heading={head}&pitch=10"
url = "https://maps.googleapis.com/maps/api/streetview?location={lat},{lon}&size=800x600&heading={head}&pitch=10"
heads=[-120,-60,0,60,120,180]
api1="&key=AIzaSyC8IdS9zMFFI6ZJgnIP8zCSdRYV310KUyU"
api2="&key=AIzaSyBErFmSxjP6KBL2IYOOYIglmJ3vL2htJtw"
api3="&key=AIzaSyDeN3OYU3e7xGiWMxDVd7e2vFN09aoLAJ4"
api4="&key=AIzaSyDJHq5FIFG7XGhqc0YUq84D1jUEvSoFaYQ"
api5="&key=AIzaSyBjKbkHJKRXXjuAVJYXEZz6ZoLmvDTQVhA"
api6="&key=AIzaSyAohSuRcIwnJuo7ACQj2eFzoBf4yQzOc5k"
api7="&key=AIzaSyDjUlngvOCctypfHwAxX70eK8yUsOUbJWc"
api8="&key=AIzaSyD_TKPPC44D23i36BzWW3vKD0KM9BMjHlk"
api9="&key=AIzaSyAGTHMRJVjA0x8L0gVpfW7fXMGjZ6PsyMQ"
api10="&key=AIzaSyCpbHxeozbGxR8FJdjumThiTmVoT1axJ1Q"
api11="&key=AIzaSyAlswcNyabDasGKNYaoEZOxfCI-fcaRIiQ"
api12="&key=AlzaSyBd5-PX1HsXwivltFEHLh1cbC8V59Hkcx4"
apis_streets=[api1,api2,api3,api4,api5,api6,api7,api8,api9,api10,api11,api12,]
suspected_home_loc_df=pd.read_csv("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/suspected_home_loc.csv",
                                  sep="\t",index_col=0)

proxy = ""  # In case you need a proxy
proxies = {"http": proxy, "https": proxy}
sesion = requests.Session()
sesion.proxies = proxies

def crea_url(datos):
    return url.format(**datos)

def crea_url_metadata(datos):
    return url_metadata.format(**datos)

def save_imagen(datos,api):
    url = crea_url(datos)
    url=url+api
    url_metadata=crea_url_metadata(datos)
    url_metadata=url_metadata+api
    r = sesion.get(url)
    r_metadata = json.loads((sesion.get(url_metadata)).content)
    fname='%s.png' % datos["nombre"]
    if r_metadata["status"]=="OK":
        real_location=r_metadata["location"]
        f=open(fname,'wb')
        f.write(r.content)
        f.close()
        return fname,real_location,r_metadata["status"]
    return fname,None,r_metadata["status"]

f_folder="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/suspects_street/"
total_nb_requests=suspected_home_loc_df.shape[0]
max_nb=24000
count=0
info=[]

nb_del=88000

for it,line in tqdm(suspected_home_loc_df[nb_del:].iterrows()):
    for head in heads:
        count+=1
        datos = {"lat": line.lat, "lon": line.lon,"head":head,
                 "nombre":f_folder+"%d_imag_%d_ang_%s"%(nb_del+it,line.id,str(head))}
        file_name,real_loc,stat=save_imagen(datos,apis_streets[count//max_nb]) 
        print(stat)
        if real_loc is None:
            info.append((file_name,None,None,stat))
        else:
            info.append((file_name,real_loc["lat"],real_loc["lng"],stat))
        if stat=="OVER_QUERY_LIMIT":
             count=min(count//max_nb+1,len(apis_streets)-1)*max_nb
        elif stat=="REQUEST_DENIED":
            print(it, apis_streets[count//max_nb])
            pause=input("What do I do? ")
            count=min(count//max_nb+1,len(apis_streets)-1)*max_nb
        if count >=len(apis_streets)*max_nb:
            time.sleep(3600*24)
            count=0

#info_df=pd.DataFrame(info,columns=["fname","lat","lon","status"])
#info_df.to_csv("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/street_info.csv")


