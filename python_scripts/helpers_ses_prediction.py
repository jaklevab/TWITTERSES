################################# IMPORTS #################################

import gensim
import gensim.models.word2vec as w2v
from tqdm import tqdm_notebook as tqdmn
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from gensim.matutils import Dense2Corpus
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
import inspect, re
#from keras import backend as K
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import SGD, RMSprop, Nadam, Adagrad, Adadelta, Adam, Adamax

################################# TODO #################################

################################# HELPERS #################################

""" Returns name of variable """
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

""" Load textual information of users """
def load_text_data(fname="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/text_ses_net_approach/cleaned_full_info.txt"):
    #dic=pickle.load(open( "/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/new_dic_real_home_max_filtered.p", "rb" ))
    #ids=[int(ix) for ix in dic.keys()]
    print("Loading Tweets ...")
    f=open(fname,"r")
    data=[(line.split()[0],line.split()[1],line.split()[2]," ".join(line.split()[3:-2]).replace('\n','')) for line in tqdmn(f.readlines())]
    tweets=pd.DataFrame(data,columns=["id","sth","sth_class","tweet"])
    tweets_id=[int(idx) for idx in tweets.id];tweets["id"]=tweets_id#;tweets=tweets[[id in ids for id in tweets.id]]
    usr_text=(tweets.groupby('id',squeeze=True,)['tweet'].apply(lambda x: "%s" % ' '.join(x))).to_frame()
    return usr_text

""" Load textual information of mutual mention net users """
def load_text_net_data(fname="/home/jlevyabi/seacabo/data_files/non_empty_undir_network_thresh_5_ids_body.txt"):
    f=open(fname,"r")
    data=[(line.split('\t')[0],line.split('\t')[1].replace('\n','')) for line in tqdmn(f.readlines())]
    tweets=pd.DataFrame(data,columns=["id","tweet"])
    tweets_id=[int(idx) for idx in tweets.id];tweets["id"]=tweets_id
    usr_text=(tweets.groupby('id',squeeze=True,)['tweet'].apply(lambda x:"%s" % ' '.join(x))).to_frame()
    return usr_text

""" Load textual information of mutual mention net users """
def load_ses_data(iris_fname='/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/geo_real_home_iris.csv',
                  insee_fname='/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/geo_real_home_insee_200m.csv'):
    #print("Loading SES info ...")
    # User to SES info for IRIS and INSEE_200m datasets
    usrs_with_iris_income=pd.read_csv(iris_fname,sep='\t')
    iris_data=usrs_with_iris_income.drop(['nom_iris','nom_region','nom_com','insee_com','index_right','iris','IRIS','COM','code_iris','code_dept','insee_com',
                                          'nom_dept','typ_iris','LIBIRIS','LIBCOM'],axis=1).groupby('usr', as_index=False)["DEC_D113","DEC_MED13","DEC_D913"]
    usrs_with_insee_income=pd.read_csv(insee_fname,sep='\t')
    insee_200_data=usrs_with_insee_income.drop(["geometry","index_right"], axis=1).groupby('usr', as_index=False)['sal_pc_aprox']
    # Associate to most frequent/avg income of the places he's been in
    get_mode = lambda x: x.value_counts(dropna=True).index[0]
    get_mode.__name__ = "most frequent"
    iris_data_avg=iris_data.mean()
    iris_data_mode=iris_data.agg(get_mode)
    insee_200_data_avg=insee_200_data.mean()
    insee_200_data_mode=insee_200_data.agg(get_mode)
    # Merge INSEE and IRIS SES information
    data_avg=pd.merge(iris_data_avg,insee_200_data_avg,on='usr',how='inner')
    data_mode=pd.merge(iris_data_mode,insee_200_data_mode,on='usr',how='inner')
    # Parse to int identities of users
    cors_id=[int(idx) for idx in data_avg.usr];data_avg["id"]=cors_id
    cors_id=[int(idx) for idx in data_mode.usr];data_mode["id"]=cors_id
    return(data_avg,data_mode)

""" Compute avg word2vec, topic distribution + Merge text info with ses info """
def ses_and_text(usr_text,cluster_dict,data_avg,data_mode,w2v_file='/home/jlevyabi/seacabo/data_files/lowe_dim_sosweet2vec.w2v'):
    usr_text=usr_text.reset_index();print("Merging SES and text...")
    usr_text['id']=[int(id_)for id_ in usr_text.id]
    data_avg['id']=[int(id_)for id_ in data_avg.id]
    data_mode['id']=[int(id_)for id_ in data_mode.id]
    model = Word2Vec.load(w2v_file)
    #
    w2v_fts=[]
    for tweet in tqdmn(usr_text.tweet):
        inter=[model[word] for word in tweet.split() if word in model]
        if len(inter)==0:
            w2v_fts.append(None)
        else:
            w2v_fts.append(np.mean(inter,axis=0))
    usr_text["avg_w2v"]=w2v_fts;print("w2v_added...",usr_text.head())
    word2cluster_only_pos={word:cluster_nb for cluster_nb,cluster_words in cluster_dict.items() for word in cluster_words}
    #
    clust_freq_only_pos=[]
    for tweet in tqdmn(usr_text.tweet):
        clust_freq_only_pos.append((Counter([word2cluster_only_pos[word] for word in tweet.split() if word in word2cluster_only_pos])))
    cfd_only_pos=[{k:(v+0.0)/(sum(dic_count.values()))for k,v in dic_count.items()} for dic_count in clust_freq_only_pos]
    usr_text["cfd"]=[np.array(list({clus:(dic_count[clus] if clus in dic_count else 0) for clus in range(len(cluster_dict))}.values())) for dic_count in cfd_only_pos]
    print("cfd_added...",usr_text.head())
    text_avg_inc=pd.merge(usr_text,data_avg,on="id",how="inner").drop("usr",axis=1)
    text_mode_inc=pd.merge(usr_text,data_mode,on="id",how="inner").drop("usr",axis=1)
    text_avg_inc['id']=[int(id_)for id_ in text_avg_inc.id]
    text_mode_inc['id']=[int(id_)for id_ in text_mode_inc.id]
    return(usr_text,text_mode_inc,text_avg_inc)

"""  Shapes feature and target data to feed it to ML (sklearn format) """
def frame_for_ml(text_inc,n2v=False,neigh_cfd=False):
    print ("Formatting data for ML....")
    text_inc=text_inc[text_inc['avg_w2v'].isnull()== False]
    if n2v:
        dim_total=text_inc['avg_w2v'].reset_index(drop=True)[0].shape[0]+text_inc['cfd'].reset_index(drop=True)[0].shape[0]+text_inc['n2v'].reset_index(drop=True)[0].shape[0]
        text_fts=np.vstack([np.hstack(sample.as_matrix()).reshape((1,dim_total)) for it,sample in (text_inc[["avg_w2v","cfd","n2v"]].iterrows())])
    elif neigh_cfd:
        dim_total=text_inc['avg_w2v'].reset_index(drop=True)[0].shape[0]+text_inc['cfd'].reset_index(drop=True)[0].shape[0]+text_inc['avg_neigh_cfd'].reset_index(drop=True)[0].shape[0]
        text_fts=np.vstack([np.hstack(sample.as_matrix()).reshape((1,dim_total)) for it,sample in (text_inc[["avg_w2v","cfd","avg_neigh_cfd"]].iterrows())])
    else:
        dim_total=text_inc['avg_w2v'].reset_index(drop=True)[0].shape[0]+text_inc['cfd'].reset_index(drop=True)[0].shape[0]
        text_fts=np.vstack([np.hstack(sample.as_matrix()).reshape((1,dim_total)) for it,sample in (text_inc[["avg_w2v","cfd"]].iterrows())])
    targets_insee=np.array(text_inc['sal_pc_aprox'])
    targets_iris_med=np.array(text_inc['DEC_MED13'])
    targets_iris_low=np.array(text_inc['DEC_D113'])
    targets_iris_high=np.array(text_inc['DEC_D913'])
    return(text_fts,targets_insee,targets_iris_med,targets_iris_low,targets_iris_high)

""" 2-class ses separation of users based on cumulative distribution """
def ses_classify(insee,iris_med,iris_low,iris_high):
    nb_class=2
    ses_class=[]
    for ses in [insee,iris_med,iris_low,iris_high]:
        sorted_income=np.sort(ses)
        N=len(sorted_income)
        users_per_class=int(N/nb_class)
        income_bins=[sorted_income[i*users_per_class] for i in range(nb_class)]
        income_bins.append(max(sorted_income))
        class_income=np.digitize(x=ses,bins=income_bins, right=False)
        class_income[class_income==nb_class+1]=nb_class
        class_income=class_income-1
        ses_class.append(class_income)
    return (ses_class)

""" Logistic Regression classification """
def log_ses_class(txt,class_income):
    X_train, X_test, y_train, y_test = train_test_split(txt, class_income, test_size=0.2,)
    logit_model=sm.Logit(y_train,X_train)
    fitted_model=logit_model.fit()
    y_pred=fitted_model.predict(X_test)>=0.5
    print(classification_report(y_test,y_pred))
    return fitted_model

