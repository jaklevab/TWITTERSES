# Imports

#%autoreload 2
import sqlite3 as lite
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import geopandas as gpd
import sys
sys.path.append('/datastore/complexnet/jlevyabi/network_representation/python_scripts/')
import helpers_ses_prediction as hsp
from tqdm import tqdm
import pickle
from tqdm import tqdm_notebook as tqdmn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tkr
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from collections import Counter
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from scipy.spatial import cKDTree
from collections import Counter
#from geopy.distance import vincentys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
import warnings
import pickle
import unidecode,re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

uk = '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 \
+x_0=400000 +y_0=-100000 +ellps=airy \
+towgs84=446.448,-125.157,542.06,0.15,0.247,0.842,-20.489 +units=m +no_defs'
from pyproj import transform,Proj

def proj_arr(points,proj_to):
    inproj = Proj(init='epsg:4326')
    outproj = Proj(proj_to)
    func = lambda x: transform(inproj,outproj,x[1],x[0])
    return np.array(list(map(func, points)))

def take_most_frequent(geopandas_usr):
    polys_visited=list(geopandas_usr.idINSPIRE)
    #time_of_visit=[datetime(row.year,row.month,row.day,row.minu,row.sec) for it,row in geopandas_usr.iterrows()]
    locat_mode=Counter(polys_visited).most_common(1)[0][0]
    idx_mode=polys_visited.index(locat_mode)
    return idx_mode,geopandas_usr.iloc[idx_mode][["lat","lon"]]

def get_check_in_rate_margin_most_freq(geopandas_usr):
    polys_visited=list(geopandas_usr.idINSPIRE)
    inter=Counter(polys_visited).most_common(2)
    if len(inter)<2:
        return None,None,None,None,None
    locat_mode,sec_locat_mode=inter
    idx_mode,idx_mode_sec=polys_visited.index(locat_mode[0]),polys_visited.index(sec_locat_mode[0])
    return (idx_mode,geopandas_usr.iloc[idx_mode][["lat","lon"]],
            idx_mode_sec,geopandas_usr.iloc[idx_mode_sec][["lat","lon"]],
           ((locat_mode[1]+0.0-sec_locat_mode[1])/(sec_locat_mode[1]+locat_mode[1])))

def take_most_frequent_night(geopandas_usr,start=21,stop=6) :
    polys_visited=(geopandas_usr.idINSPIRE)
    polys_visited_night=polys_visited[(geopandas_usr.hour>=start)|(geopandas_usr.hour<stop)]
    if len(polys_visited_night)==0:
        return None,None
    locat_mode=Counter(polys_visited_night).most_common(1)[0][0]
    idx_mode=list(polys_visited).index(locat_mode)
    return idx_mode,geopandas_usr.iloc[idx_mode][["lat","lon"]]

def get_distance_matrix(geopandas_usr):
    x = np.array(geopandas_usr[["lat","lon"]]).astype(float).tolist()
    y=proj_arr(x,uk)
    ztree = cKDTree(y)
    z = ztree.sparse_distance_matrix(ztree,1e6,p=2).todense()
    return z

def distance_to_home(geopandas_usr,select_home_loc,args):
    idx,loc=select_home_loc(geopandas_usr,*args)
    if idx is None:
        return None,None,None
    mat_dist=get_distance_matrix(geopandas_usr)
    return mat_dist[idx,:].tolist()[0],list(geopandas_usr.day),list(geopandas_usr.hour)

def go_through_home_candidates(dic_gpd,select_home_loc):
    dic_exam={}
    for usr,gpd in (dic_gpd.items()):
        idx,loc=select_home_loc(gpd)
        if idx is None:
            continue
        dic_exam.setdefault(usr,gpd.iloc[idx])
    return dic_exam

def go_through_geol_users(dic_gpd,select_home_loc,args,outlier_lim=6e4):
    dic_per_day={k:np.zeros(24) for k in range(7)}
    dic_nb_per_day={k:np.zeros(24) for k in range(7)}
    dic_exam={}
    loss=[]
    for usr,gpd in (dic_gpd.items()):
        dic_exam.setdefault(usr,[])
        dists,days,hours=distance_to_home(gpd,select_home_loc,args)
        new_dists=np.array(dists)
        if dists is None:
            continue
        loss.append(1-(np.sum([new_dists<outlier_lim])+0.0)/len(dists) )
        dists=new_dists[new_dists<outlier_lim]
        for dist,day,hour in zip(dists,days,hours):
            dic_exam[usr].append(dist)
            dic_per_day[day][hour]+=dist
            dic_nb_per_day[day][hour]+=1
    dic_day={}
    for k,v in dic_per_day.items():
        dic_day[k]=(v/dic_nb_per_day[k])/100
    return dic_day,dic_exam,loss

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
    return (df_tweets)


print("Loaded functions")

dic_final_not_nan=pickle.load(open("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_dic.p","rb"))
dic_iris_not_nan=pickle.load(open("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_dic_iris.p","rb"))

print("Pickled loaded")

home_most_freq_all=go_through_home_candidates(dic_final_not_nan,take_most_frequent)
home_most_freq_night=go_through_home_candidates(dic_final_not_nan,take_most_frequent_night)
home_most_freq_all_iris=go_through_home_candidates(dic_iris_not_nan,take_most_frequent)
home_most_freq_night_iris=go_through_home_candidates(dic_iris_not_nan,take_most_frequent_night)

print("Number of geolocated users INSEE(most freq) ... %d"%len(home_most_freq_all))
print("Number of geolocated users INSEE(most freq night) ... %d"%len(home_most_freq_night))
print("Number of geolocated users IRIS(most freq) ... %d"%len(home_most_freq_all_iris))
print("Number of geolocated users INSEE(most freq night) ... %d"%len(home_most_freq_night_iris))
print("Number of geolocated users with INSEE and IRIS info ...%d"%
      len(set(list(dic_final_not_nan.keys())).intersection(set(list(dic_iris_not_nan.keys())))))

usr2insee=pd.DataFrame([x.values for x in home_most_freq_all.values()], columns=(list(home_most_freq_all.values())[0]).keys())
usr2insee["poor_men"]=(usr2insee.men_prop/usr2insee.men)
usr2iris=pd.DataFrame([x.values for x in home_most_freq_all_iris.values()],
                      columns=(list(home_most_freq_all_iris.values())[0]).keys())

#Bio text data
f_accounts="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles.p"
f_deleted="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles_deleted_accounts.p"
dic_bio=pickle.load(open(f_accounts,"rb"))
dic_bio_deleted=pickle.load(open(f_deleted,"rb"))

print("Bio text data")
#N-Gram vectorizer
french_stopwords = list(set(stopwords.words('french')))
eng_stopwords = list(set(stopwords.words('english')))
n_grams_bio_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,
                                 max_features=450,ngram_range=(1,2),
                                lowercase=True)

print("N-Gram vectorizer")
#Clean profile info
tweet_clean = lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",
                                       unidecode.unidecode(x.replace("_",""))).split())


print("Clean profile info")
#Generate profile info dataframe
cols_of_interest=["id","followers_count","friends_count",
                         "listed_count","favourites_count","statuses_count","description"]
profile_data=[[usr[0]._json[k] for k in cols_of_interest] for usr in dic_bio ]
df_profile_data=pd.DataFrame(profile_data,columns=cols_of_interest)
df_profile_data["description"]=df_profile_data.description.apply(tweet_clean)
n_grams_bio=n_grams_bio_vect.fit_transform(list(df_profile_data.description.values))

usr_tweet_text=pd.read_csv("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/tweets/all_geolocated_users.csv",
    sep=';',header=0,)
usr_text=(usr_tweet_text.dropna(how="any").drop(["tweet_id","tweet_date"],axis=1).groupby(
    'user_id',squeeze=True,)['tweet_text'].apply(lambda x: "%s" % ' '.join(x))).to_frame()
usr_text.reset_index(inplace=True)

print("usr_text loaded")
n_grams_tweet_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,
                                 max_features=560,ngram_range=(1,1),
                                lowercase=True)
n_grams_tweet=n_grams_tweet_vect.fit_transform(list(usr_text.tweet_text.values))
d200=pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_200.p","rb"))
usr_text=get_cluster_info(d200,usr_text);

df_usr_profile_tweets=pd.merge(df_profile_data,usr_text,left_on="id",right_on="user_id")

print("df_usr_profile_tweets computed")
#Profile Information: Shallow features
mat_shallow_bio=np.vstack([np.hstack(sample.as_matrix()).reshape((1,5))
                            for it,sample in (df_usr_profile_tweets[["followers_count","friends_count",
                                                                     "listed_count",
                                                                     "favourites_count",
                                                                     "statuses_count",]].iterrows())])
#Profile Information: N-grams
n_grams_bio_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,max_features=450,ngram_range=(1,2),lowercase=True)
mat_n_grams_bio=n_grams_bio_vect.fit_transform(list(df_usr_profile_tweets.description.values)).todense()
#Tweet Information: N-grams
n_grams_tweet_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,max_features=560,ngram_range=(1,1),lowercase=True)
mat_n_grams_tweet=n_grams_tweet_vect.fit_transform(list(df_usr_profile_tweets.tweet_text.values)).todense()
#Tweet Information: Topics
mat_topics_tweet=np.vstack([np.hstack(sample.as_matrix()).reshape((1,200))
                            for it,sample in (df_usr_profile_tweets[["cfd_200",]].iterrows())])
# All non-SES info
data_matrix=np.hstack([mat_n_grams_tweet,mat_n_grams_bio,mat_topics_tweet,mat_shallow_bio])
df_usr_profile_tweets["fts"]=[row for row in data_matrix.tolist()]
ses_text_insee=pd.merge(df_usr_profile_tweets,usr2insee,left_on="id",right_on="usr")
ses_text_iris=pd.merge(df_usr_profile_tweets,usr2iris,left_on="id",right_on="usr")

#Node2vec
print(ses_text_iris.iloc[0].values)
import gensim;model_jwalk=gensim.models.Word2Vec.load('/home/jlevyabi/seacabo/data_files/jwalk.emb')
jwalk_id_nets=[int(x) for x in ses_text_iris.usr if str(x)+".0" in model_jwalk.wv.vocab]
inter_jwalk=[model_jwalk.wv[str(x)+".0"] for x in jwalk_id_nets]

jwalk_data=pd.DataFrame()
jwalk_data["id"]=jwalk_id_nets
jwalk_data["n2v"]=inter_jwalk

ses_text_n2v_iris=pd.merge(ses_text_iris,jwalk_data,left_on="usr",right_on="id")
ses_iris_n2v_class_try=np.array(ses_text_n2v_iris.DEC_D913>np.mean(ses_text_n2v_iris.DEC_D913)).astype(np.int)# 2 class
mat_info_fts=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_n2v_iris.iloc[0]["fts"])))
                            for it,sample in (ses_text_n2v_iris[["fts",]].iterrows())])

inter=[]
for row_1,row_2 in zip(ses_text_n2v_iris.fts,ses_text_n2v_iris.n2v):
    inter.append(np.hstack([np.array(row_1),np.array(row_2)]))

ses_text_n2v_iris["n2v+fts"]=inter
mat_info_n2v_fts=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text_n2v_iris.iloc[0]["n2v+fts"])))
                            for it,sample in (ses_text_n2v_iris[["n2v+fts",]].iterrows())])

print("SES + text data loaded")

names = [ "Random Forest", "Neural Net 1","Neural Net 2","Neural Net 3", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    MLPClassifier(hidden_layer_sizes=(100,100),alpha=1),
    MLPClassifier(hidden_layer_sizes=(100,100,100),alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# Clear display
print(chr(27) + "[2J")
print("Classifying dataset no network")
X_fts = StandardScaler().fit_transform(mat_info_fts)
for name, clf in zip(names, classifiers):
    score=cross_val_score(clf, X_fts, y=ses_iris_n2v_class_try, cv=10)
    print ("2-Classification accuracy for %s:  %s"%(name,str(score)))

print("Classifying dataset network")
X_fts_n2v = StandardScaler().fit_transform(mat_info_n2v_fts)
for name, clf in zip(names, classifiers):
    score=cross_val_score(clf, X_fts_n2v, y=ses_iris_n2v_class_try, cv=10)
    print ("2-Classification accuracy for %s:  %s"%(name,str(score)))

