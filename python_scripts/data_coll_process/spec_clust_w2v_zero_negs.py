import argparse
import sys
import gensim.models.word2vec as w2v
from gensim.models import Word2Vec
from gensim.similarities import MatrixSimilarity
from gensim.matutils import Dense2Corpus
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gensim
from tqdm import tqdm
from sklearn.cluster import spectral_clustering,SpectralClustering
import pickle

""" Clean individual words + filter out words not appearing more than min_times through all sosweet database"""
def after_treatment_word2vec(w2v_model,min_times):
    treated_words=[]
    for word in w2v_model.wv.vocab:
        alpha_wrds=''.join(x for x in word if x.isalpha())
        if len(alpha_wrds)==len(word) and w2v_model.wv.vocab[word].count>min_times:
            treated_words.append(word)
    return treated_words

""" Generate spectral clusters based on w2v embeddings"""
def w2v_spectral_clustering(w2v_model,f_out,nb_clusts,min_times=70,threshold_for_bug = 0.000000001,):
    vocab = after_treatment_word2vec(model,min_times)
    print("Nb words in vocabulary.... %d"%len(vocab))
    print("Computing similarity matrix .....")
    modelvect = model[vocab]
    A_sparse = sparse.csr_matrix(np.array([gensim.matutils.unitvec(i) for i in tqdm(modelvect)]))
    similarities = cosine_similarity(A_sparse,dense_output=True)
    similarities[(similarities)<threshold_for_bug]= threshold_for_bug #explained in the paper
    print("Spectral Clustering ....")
    spectral = SpectralClustering(n_clusters=nb_clusts,affinity="precomputed",n_jobs=30)
    spectral.fit(similarities)
    print("SC labeling .... ")
    labels = spectral.fit_predict(similarities)
    word_clusters={i:[] for i in range(nb_clusts)}
    for it,lab in enumerate(labels):
        word_clusters[lab].append(vocab[it])
    print("Saving Clusters ..... ")
    with open(f_out, 'wb') as f:
    	pickle.dump(word_clusters,f)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-w2v", "--w2v_input")
	parser.add_argument("-out", "--output")
	parser.add_argument("-nbc", "--nb_clusters")
    w2v_input, f_out, nb_clusters = args.w2v_input, args.output, args.nb_clusters
	args = parser.parse_args()
    print("Model Loading....")
    w2v_model = Word2Vec.load(w2v_input)
    w2v_spectral_clustering(w2v_model,f_out,nb_clusters)
    print("Done ....!")
