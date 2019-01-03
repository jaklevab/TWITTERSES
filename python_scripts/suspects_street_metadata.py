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
    os.system("rm /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/street_info.csv")

waiting=int(input("Write hours to wait before starting collection "))
time.sleep(3600*waiting)

# URLs antiguas
url_metadata="https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&size=800x600&heading={head}&pitch=10"
heads=[-120,-60,0,60,120,180]
api="&key=AIzaSyC8IdS9zMFFI6ZJgnIP8zCSdRYV310KUyU"
suspected_home_loc_df=pd.read_csv("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/suspected_home_loc.csv",
                                  sep="\t",index_col=0)

proxy = ""  # In case you need a proxy
proxies = {"http": proxy, "https": proxy}
sesion = requests.Session()
sesion.proxies = proxies


def crea_url_metadata(datos):
    return url_metadata.format(**datos)

def save_image_metadata(datos,api):
    url_metadata=crea_url_metadata(datos)
    url_metadata=url_metadata+api
    r_metadata = json.loads((sesion.get(url_metadata)).content)
    fname='%s.png' % datos["nombre"]
    if r_metadata["status"]=="OK":
        real_location=r_metadata["location"]
        return fname,real_location,r_metadata["status"]
    return fname,None,r_metadata["status"]

f_folder="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/suspects_street/"
total_nb_requests=suspected_home_loc_df.shape[0]
max_nb=24000
count=0
info=[]

for it,line in tqdm(suspected_home_loc_df.iterrows()):
    for head in heads:
        count+=1
        datos = {"lat": line.lat, "lon": line.lon,"head":head,
                 "nombre":f_folder+"%d_imag_%d_ang_%s"%(it,line.id,str(head))}
        file_name,real_loc,stat=save_image_metadata(datos,api) 
        print(stat)
        if real_loc is None:
            info.append((file_name,None,None,stat))
        else:
            info.append((file_name,real_loc["lat"],real_loc["lng"],stat))

info_df=pd.DataFrame(info,columns=["fname","lat","lon","status"])
info_df.to_csv("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/street_info.csv")


