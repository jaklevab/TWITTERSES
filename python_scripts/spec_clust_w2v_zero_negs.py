
print("Imports....")
import gensim.models.word2vec as w2v
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from gensim.matutils import Dense2Corpus
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from tqdm import tqdm

print("Model Loading....")
model = Word2Vec.load('/home/jlevyabi/seacabo/data_files/lowe_dim_sosweet2vec.w2v')

print(model.most_similar('mcdo',topn=10))
def after_treatment_word2vec(w2v_model,min_times):
    treated_words=[]
    for word in w2v_model.wv.vocab:
        alpha_wrds=''.join(x for x in word if x.isalpha())
        if len(alpha_wrds)==len(word) and w2v_model.wv.vocab[word].count>min_times:
            treated_words.append(word)
    return treated_words

vocab = after_treatment_word2vec(model,min_times=70)
print("Nb words in vocabulary.... %d"%len(vocab))

print("Computing similarity matrix .....")
modelvect = model[vocab]
A_sparse = sparse.csr_matrix(np.array([gensim.matutils.unitvec(i) for i in tqdm(modelvect)]))
similarities = cosine_similarity(A_sparse,dense_output=True)
threshold_for_bug = 0.000000001
similarities[(similarities)<threshold_for_bug]= threshold_for_bug
nb_clusts=100

print("Spectral Clustering ....")
from sklearn.cluster import spectral_clustering,SpectralClustering
spectral = SpectralClustering(n_clusters=nb_clusts,affinity="precomputed",n_jobs=30)
spectral.fit(similarities)

print("SC labeling .... ")
labels = spectral.fit_predict(similarities)
word_clusters={i:[] for i in range(nb_clusts)}
for it,lab in enumerate(labels):
    word_clusters[lab].append(vocab[it])

print("Saving Clusters ..... ")
import pickle
#phrunch_clusters=pickle.load(open('/home/jlevyabi/seacabo/data_files/spec_clusters.p','rb'))
with open('/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/spec_corrected_clusters_only_pos_entries.p', 'wb') as f:
	pickle.dump(word_clusters,f)

print("Done ....!")
