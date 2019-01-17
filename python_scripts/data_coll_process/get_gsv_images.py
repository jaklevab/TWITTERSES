# coding:utf-8
from __future__ import print_function
from __future__ import division
import requests
from tqdm import tqdm
import pandas as pd
import time
import json
import sys,os
import argparse
from os import listdir
from os.path import isfile, join

waiting=int(input("Write hours to wait before starting collection "))
time.sleep(3600*waiting)
sins_cleaninsing = input("Remove previous ?")

proxy = ""  # In case you need a proxy
proxies = {"http": proxy, "https": proxy}
sesion = requests.Session()
sesion.proxies = proxies
shot_angles = [-120,-60,0,60,120,180]

""" Formatting data for url """
def crea_url(datos):
    return url.format(**datos)

""" Formatting metadata for url """
def crea_url_metadata(datos):
    return url_metadata.format(**datos)

""" Remove older files contained in storing directory """
def cleanse_sins(image_folder,log_file,gsv_metadata):
    os.system("rm -r %s"%image_folder)
    os.system("mkdir %s"%image_folder)
    os.system("rm %s"%log_file)
    os.system("touch %s"%log_file)
    os.system("rm %s"%gsv_metadata)
    os.system("touch %s"%gsv_metadata)

""" Crawl gsv image and metadata"""
def save_imagen(datos,api,log_file):
    url = crea_url(datos)
    url_metadata=crea_url_metadata(datos)
    url=url+api
    url_metadata=url_metadata+api
    r = sesion.get(url)
    r_metadata = json.loads((sesion.get(url_metadata)).content)
    fname='%s.png' % datos["nombre"]
    if r_metadata["status"]=="OK":
        real_location=r_metadata["location"]
        with open(fname,'wb') as f:
            f.write(r.content)
        with open(log_file, "a") as log_f:
            log_f.write("%s;%s;True"%(r_metadata["status"],fname))
        return fname,real_location,r_metadata["status"]
    else:
        with open(log_file, "a") as log_f:
            log_f.write("%s;%s,False"%(r_metadata["status"],fname))
    return fname,None,r_metadata["status"]

""" Returns images from original dataset not yet downloaded """
def check_images_prev_downloaded(image_folder,log_file,df_coords_to_sample):
    data_log = pd.read_csv(log_file,header=None,names=["file","status","verdad"])
    nb_images = df_coords_to_sample.shape[0]
    downloaded_ims = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
    to_download_ims = ["imag_%d_ang_%s"%(idx,str(head)) for idx in range(nb_images) for head in shot_angles ]
    inexistent_ims = [line.file for it,line in data_log.iterrows() if not line.verdad]
    remaining_ims = list(set(to_download_ims)-set(downloaded_ims)-set(inexistent_ims))
    print("# images to download %d\n # images already downloaded %d\n # images to download %d"%(
        len(to_download_ims),len(downloaded_ims),len(remaining_ims)))
    print (remaining_ims[:100])
    return list(set([int(x.split("_")[1]) for x in remaining_ims]))

""" Crawl list of of coordinates """
def get_images(data_lat_lon, f_folder, log_file,list_of_apis,gsv_metadata_file):
    print("Checking previous gsv collections")
    images_not_downloaded_yet = check_images_prev_downloaded(f_folder,log_file,data_lat_lon)
    use_count_per_api = {api: 0 for api in list_of_apis}
    api_idx = 0
    curr_api = list_of_apis[api_idx]
    launch=(input("Launch? "))
    if launch !="y":
        sys.exit("Aborted")
    for it,line in tqdm(data_lat_lon.iterrows()):
        if not(it in images_not_downloaded_yet):
            continue
        for head in shot_angles:
            use_count_per_api[curr_api]+=1
            datos = {"lat": line.lat, "lon": line.lon,"head":head,"nombre":f_folder+"imag_%d_ang_%s"%(it,str(head))}
            file_name,real_loc,stat=save_imagen(datos,curr_api,log_file)
            print(stat,file_name)
            if real_loc is None:
                info=[(file_name,None,None,stat),]
            else:
                info=[(file_name,real_loc["lat"], real_loc["lng"],stat),]
            with open(gsv_metadata_file, 'a') as f:
                df_info = pd.DataFrame(info,columns=["name","lat","lon","status"])
                df_info.to_csv(f, header=(it==0))
            if stat=="OVER_QUERY_LIMIT":
                api_idx+=1
                curr_api = list_of_apis[api_idx]
            elif stat=="REQUEST_DENIED":
                print(curr_api,use_count_per_api[curr_api])
                sys.exit("REQUEST_DENIED: Check again what to do")
            if api_idx >=len(apis_streets):
                time.sleep(3600*24)
                api_idx=0
                curr_api = list_of_apis[api_idx]
    return

if __name__ == "__main__" :
    print("Parsing Arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',help = 'input of lat,long sampled coordinates to get',
                  default="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/suspected_home_loc.csv")
    parser.add_argument('-of', '--output_folder',help = 'folder for output',
                    default="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/images_suspected_locs/suspects_street/")
    parser.add_argument('-gsv_met', '--gsv_metadata',help = 'file for adding metadata returned by gsv',
                    default="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/images_suspected_locs/suspects_street/gsv_metadata.csv")
    parser.add_argument('-log', '--log_file',help = 'log file',
                    default="/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/images_suspected_locs/suspects_street/gsv_crawl.log")
    args = parser.parse_args()
    #
    print("Reading Files....")
    url_metadata="https://maps.googleapis.com/maps/api/streetview/metadata?location={lat},{lon}&size=800x600&heading={head}&pitch=10"
    url = "https://maps.googleapis.com/maps/api/streetview?location={lat},{lon}&size=800x600&heading={head}&pitch=10"
    api1="&key=" # to fill
    apis_streets=[api1,]
    #
    data_input_lon_lat = pd.read_csv(args.input,header=0)
    output_folder = args.output_folder
    if not output_folder.endswith("/"):
        output_folder+="/"
    #
    if sins_cleaninsing=="y":
        cleanse_sins(output_folder,args.log_file,args.gsv_metadata)
    #
    print("Launching...")
    get_images(data_input_lon_lat, output_folder, args.log_file,apis_streets,args.gsv_metadata)
