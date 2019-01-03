# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdmn
from geopandas import GeoDataFrame
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as tkr
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from geopandas import sjoin
import pickle
import re;pat = re.compile(r'''(-*\d+\.\d+ -*\d+\.\d+);*''');new_geo=[]
import seaborn as sns
import sys
import sqlite3 as lite
sys.path.append('/warehouse/COMPLEXNET/jlevyabi/network_representation/python_scripts/')
import helpers_ses_prediction as hsp
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime
from collections import Counter
from scipy.spatial import cKDTree
from collections import Counter
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
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")
import pickle
import unidecode,re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
author=input("Enter Name of the architect for which to predict (Janos or Matyas): ")

# Architects Answer
data_answer_janos=pd.read_excel("/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/bigsample_Janos_answer.xlsx",
names=["Index","SES_answer","confidence",])
data_answer_janos["comments"]=np.nan
data_answer_matyas=pd.read_excel("/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/images_suspected_locs/bigsample_Matyas_answer.xlsx",
names=["Index","untreated_SES_answer","confidence","comments"])

# Architects Sample Data
full_archi_sample_df=pd.read_pickle("/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/full_archi_sample.p")
archis=full_archi_sample_df.round({'lat': 2, 'lon': 2}).drop_duplicates(subset=["lat","lon"]).index
full_archi_sample_df_no_dup=full_archi_sample_df.iloc[archis]
ang_list=["tohtml_street_ang_%s"%str(ang) for ang in [-120,-60,0,60,120,180]]

cols=["Index","id","Latitude","Longitude","Nom_Com","Weekday","Hour"]
y=[]
j=0
for it,item in full_archi_sample_df_no_dup[["id","lat","lon","weekday","hour","nom_com","tohtml_zm17","tohtml_zm18","tohtml_zm19"]+ang_list].iterrows():
    if item.values[-1] and not (None in item.values) and item.values[-2]!=item.values[-3]:
        item["lat"]=np.round(item["lat"],5)
        item["lon"]=np.round(item["lon"],5)
        x=(j,item.id,item.lat,item.lon,item.nom_com,item.weekday,item.hour)
        j+=1
        y.append(x)

n_tot=1000
sample_delivered=y[:n_tot]
data_sample_janos=pd.DataFrame(sample_delivered[:667],columns=cols)
data_janos=pd.merge(data_answer_janos,data_sample_janos,on="Index")
data_janos["author"]="janos"
data_sample_matyas=pd.DataFrame(sample_delivered[333:],columns=cols)
data_matyas=pd.merge(data_answer_matyas,data_sample_matyas,on="Index")
data_matyas["author"]="matyas"

# Profile n-grams

#Bio text data
f_accounts="/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles.p"
f_deleted="/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles_deleted_accounts.p"
dic_bio=pickle.load(open(f_accounts,"rb"))
dic_bio_deleted=pickle.load(open(f_deleted,"rb"))

#N-Gram vectorizer
print("Bio text data")
french_stopwords = list(set(stopwords.words('french')))
eng_stopwords = list(set(stopwords.words('english')))
n_grams_bio_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,
                                 max_features=450,ngram_range=(1,2),
                                lowercase=True)

#Clean profile info
print("N-Gram vectorizer")
tweet_clean = lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",unidecode.unidecode(x.replace("_",""))).split())

#Generate profile info dataframe
print("Clean profile info")
cols_of_interest=["id","followers_count","friends_count","listed_count","favourites_count","statuses_count","description"]
profile_data=[[usr[0]._json[k] for k in cols_of_interest] for usr in dic_bio ]
df_profile_data=pd.DataFrame(profile_data,columns=cols_of_interest)
df_profile_data["description"]=df_profile_data.description.apply(tweet_clean)
n_grams_bio=n_grams_bio_vect.fit_transform(list(df_profile_data.description.values))

# Tweets Data
usr_tweet_text=pd.read_csv("/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/tweets/all_geolocated_users.csv",sep=';',header=0,)
usr_text=(usr_tweet_text.dropna(how="any").drop(["tweet_id","tweet_date"],axis=1).groupby(
    'user_id',squeeze=True,)['tweet_text'].apply(lambda x: "%s" % ' '.join(x))).to_frame()
usr_text.reset_index(inplace=True)

## Tweets data : N-Grams
print("usr_text loaded")
n_grams_tweet_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords, max_features=560,ngram_range=(1,1),lowercase=True)
n_grams_tweet=n_grams_tweet_vect.fit_transform(list(usr_text.tweet_text.values))

## Tweets data : Semantical Features
def get_cluster_info(dic_clus,df_tweets):
    nb_clusters=len(list(dic_clus.keys()))
    word2cluster_only_pos={word:cluster_nb for cluster_nb,cluster_words in dic_clus.items() for word in cluster_words}
    clust_freq_only_pos=[]
    for tweet in tqdm(df_tweets.tweet_text):
        clust_freq_only_pos.append((Counter([word2cluster_only_pos[word] for word in tweet.split() if word in word2cluster_only_pos])))
    cfd_only_pos=[{k:(v+0.0)/(sum(dic_count.values()))for k,v in dic_count.items()} for dic_count in clust_freq_only_pos]
    df_tweets["cfd_%d"%nb_clusters]=[np.array(list({clus:(dic_count[clus] if clus in dic_count else 0) for clus in range(len(dic_clus))}.values())) for dic_count in cfd_only_pos]
    return (df_tweets)

from tqdm import tqdm
d100=pickle.load(open("/home/jlevyabi/seacabo/data_files/spec_corrected_clusters_only_pos_entries_100.p","rb"))
usr_text=get_cluster_info(d100,usr_text);

# Merge profiles and tweets

df_usr_profile_tweets=pd.merge(df_profile_data,usr_text,left_on="id",right_on="user_id")
#Profile Information: Shallow features
mat_shallow_bio=np.vstack([np.hstack(sample.as_matrix()).reshape((1,5))
                            for it,sample in (df_usr_profile_tweets[["followers_count","friends_count", "listed_count",
                                                                     "favourites_count","statuses_count",]].iterrows())])
#Profile Information: N-grams
n_grams_bio_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,
                                 max_features=450,ngram_range=(1,2),lowercase=True)
mat_n_grams_bio=n_grams_bio_vect.fit_transform(list(df_usr_profile_tweets.description.values)).todense()

#Tweet Information: N-grams
n_grams_tweet_vect=CountVectorizer(stop_words=french_stopwords+eng_stopwords,
                                 max_features=560,ngram_range=(1,1),lowercase=True)
mat_n_grams_tweet=n_grams_tweet_vect.fit_transform(list(df_usr_profile_tweets.tweet_text.values)).todense()

#Tweet Information: Topics
mat_topics_tweet=np.vstack([np.hstack(sample.as_matrix()).reshape((1,100)) for it,sample in (df_usr_profile_tweets[["cfd_100",]].iterrows())])

# All non-SES info
data_matrix=np.hstack([mat_n_grams_tweet,mat_n_grams_bio,mat_topics_tweet,mat_shallow_bio])

# Everything combined syntheticly
df_usr_profile_tweets["fts"]=[row for row in data_matrix.tolist()]
df_usr_profile_tweets.drop(["followers_count","friends_count","listed_count","favourites_count","statuses_count","description","user_id","cfd_100"],inplace=True,axis=1)
df_usr_profile_tweets_janos=pd.merge(df_usr_profile_tweets,data_janos,on="id")
df_usr_profile_tweets_matyas=pd.merge(df_usr_profile_tweets,data_matyas,on="id")

# Inference Algorithms

## Janos:
if author=="Janos":
    print ("Janos Results")
    ses_text=df_usr_profile_tweets_janos
    mat_info=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text.iloc[0]["fts"]))) for it,sample in (ses_text[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    classes=np.array(ses_text.SES_answer)
    classes-=min(np.array(ses_text.SES_answer))
    #
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net 1","Neural Net 2","Neural Net 3",
              "AdaBoost","Naive Bayes", "QDA"]
    #
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        MLPClassifier(hidden_layer_sizes=(100,100),alpha=1),
        MLPClassifier(hidden_layer_sizes=(100,100,100),alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    #
    print("Classifying dataset network")
    for name, clf in zip(names, classifiers):
        score=cross_val_score(clf, X, y=classes, cv=10,n_jobs=5,verbose=1)
        print ("%d-Classification accuracy for %s:  max=%s, avg=%s"%(len(set(classes)),name,str(np.max(score)),str(np.mean(score))))
    #
elif author=="Matyas":
    ## Matyas:
    print ("Matyas Results")
    ses_text=df_usr_profile_tweets_matyas
    mat_info=np.vstack([np.hstack(sample.as_matrix()).reshape((1,len(ses_text.iloc[0]["fts"]))) for it,sample in (ses_text[["fts",]].iterrows())])
    X = StandardScaler().fit_transform(mat_info)
    classes=np.array(ses_text.untreated_SES_answer)
    classes-=min(np.array(ses_text.untreated_SES_answer))
    #
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net 1","Neural Net 2","Neural Net 3",
              "AdaBoost","Naive Bayes", "QDA"]
    #
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        MLPClassifier(hidden_layer_sizes=(100,100),alpha=1),
        MLPClassifier(hidden_layer_sizes=(100,100,100),alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    #
    print("Classifying dataset network")
    for name, clf in zip(names, classifiers):
        score=cross_val_score(clf, X, y=classes, cv=10,n_jobs=5,verbose=1)
        print ("%d-Classification accuracy for %s:  max=%s, avg=%s"%(len(set(classes)),name,str(np.max(score)),str(np.mean(score))))

