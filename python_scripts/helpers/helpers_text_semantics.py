import json
from collections import Counter
import pandas as pd
import numpy as np
import unidecode
import re
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle

french_stopwords = list(set(stopwords.words('french')))
eng_stopwords = list(set(stopwords.words('english')))

""" Extract topic proportions per users given a clustered word embedding"""
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
    return df_tweets


""" Reads profile information and vectorizes using tf-idf"""
def generate_profile_information(fbase = "/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/UKSOC_rep/"):
    #Bio text data
    dic_bio=pickle.load(open(fbase+"all_together_profiles.p","rb"))
    #Clean profile info
    tweet_clean = lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",unidecode.unidecode(x.replace("_",""))).split())
    #Generate profile info dataframe
    cols_of_interest=["id","followers_count","friends_count","listed_count","favourites_count","statuses_count","description"]
    profile_data=[[usr[0]._json[k] for k in cols_of_interest] for usr in dic_bio ]
    df_profile_data=pd.DataFrame(profile_data,columns=cols_of_interest)
    df_profile_data["description"]=df_profile_data.description.apply(tweet_clean)
    return df_profile_data

""" Synthetize sematics features for tweets (topics)"""
def generate_tw_semantic_info(geolocated_tweets,word2topics_dic):
    usr_tweet_text=pd.read_csv(geolocated_tweets,sep=';',header=0,)
    usr_text=(usr_tweet_text.dropna(how="any").drop(
        ["tweet_id","tweet_date"],axis=1).groupby('user_id',squeeze=True,)['tweet_text'].apply(lambda x: "%s" % ' '.join(x))).to_frame()
    nbtweets=(usr_tweet_text.dropna(how="any").drop(["tweet_id","tweet_date"],axis=1).groupby(
        'user_id',squeeze=True,)['tweet_text'].apply(lambda x: len(x))).to_frame().values
    usr_text["nb_tweets"]=[y[0] for y in nbtweets]
    usr_text.reset_index(inplace=True,drop=False)
    usr_text=get_cluster_info(word2topics_dic,usr_text);
    return usr_text

""" Generates all features for learning models  """
def generate_full_features(df_usr_profile_tweets,min_tweets,max_prof_fts = 450,max_tweets_fts = 560,nb_topics = 100):
    #Profile Information: Shallow features
    shallow_fts = ["followers_count","friends_count","listed_count", "favourites_count","statuses_count"]
    mat_shallow_bio=np.vstack([np.hstack(sample.as_matrix()).reshape((1,5))
                                for it,sample in (df_usr_profile_tweets[shallow_fts].iterrows())])
    #Profile Information: N-grams
    n_grams_bio_vect=TfidfVectorizer(stop_words=french_stopwords+eng_stopwords, max_features=max_prof_fts,ngram_range=(1,2),lowercase=True)
    mat_n_grams_bio=n_grams_bio_vect.fit_transform(list(df_usr_profile_tweets.description.values)).todense()
    #Tweet Information: N-grams
    n_grams_tweet_vect=TfidfVectorizer(stop_words=french_stopwords+eng_stopwords, max_features=560,ngram_range=(1,1),lowercase=True)
    mat_n_grams_tweet=n_grams_tweet_vect.fit_transform(list(df_usr_profile_tweets.tweet_text.values)).todense()
    #Tweet Information: Topics
    mat_topics_tweet=np.vstack([np.hstack(sample.as_matrix()).reshape((1,nb_topics)) for it,sample in (df_usr_profile_tweets[["cfd_%d"%nb_topics,]].iterrows())])
    # All non-SES info
    data_matrix=np.hstack([mat_n_grams_tweet,mat_n_grams_bio,mat_topics_tweet,mat_shallow_bio])
    df_usr_profile_tweets["fts"]=[row for row in data_matrix.tolist()]
    df_usr_profile_tweets=df_usr_profile_tweets[df_usr_profile_tweets.nb_tweets>min_tweets]
    return df_usr_profile_tweets
