# -*- coding: utf8 -*-
# Taken from https://github.com/dougct/mob-net/blob/master/infotheory.py

import math
import numpy as np
import pandas as pd
import scipy.stats
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt

""" Computes entropy of label distribution. """
def entropy(labels, base=2):
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * math.log(i, base)
    return ent

# Input a pandas series
def entropy_scipy(data, base=2):
        p_data= data.value_counts()/len(data) # calculates the probabilities
        entropy=scipy.stats.entropy(p_data, base=base)    # input probabilities to get the entropy
        return entropy

def compute_pi_fano(user_entropy, nr_regions):
        if user_entropy == 0.0 or nr_regions <= 1:
                return 1.0
        for p in np.arange(0.001, 1.000, 0.001):
                #print(p, user_entropy, nr_regions)
                tmp = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
                pfano = tmp + (1 - p) * math.log2(nr_regions - 1) - user_entropy
                if pfano <= 0.001:
                        return p
        return 0.0

def compute_entropies(df, region_column_name):
        entropies = defaultdict(float)
        limits = defaultdict(float)
        for i, user_checkins in df.groupby(['user_id']):
                visited_regions = list(user_checkins[region_column_name])
                # Remove unkown regions form the list.
                visited_regions[:] = [item for item in visited_regions if item != -1]
                user_entropy = entropy(visited_regions)
                entropies[i] = user_entropy
                user_prediction_limit = compute_pi_fano(user_entropy, len(visited_regions))
                limits[i] = user_prediction_limit
        return entropies, limits

def binary_entropy(nr_labels, nr_target_labels, base=2):
        p = nr_target_labels / nr_labels
        if p == 1.0 or p == 0.0:
                return 0.0
        return -p * math.log(p, base) - (1 - p) * math.log(1 - p, base)

def compute_binary_entropies(df, top_n_locations, region_column_name):
        entropies = defaultdict(float)
        limits = defaultdict(float)
        total_users = 0
        nr_users = 0
        for i, user_checkins in df.groupby(['user_id']):
                visited_regions = list(user_checkins[region_column_name])
                total_users = total_users + 1
                nr_users = nr_users + 1
                # Get the n most visited regions.
                top_n = [item for item, _ in Counter(visited_regions).most_common(top_n_locations)]
                # Remove unkown regions form the list.
                visited_regions[:] = [item for item in visited_regions if item != -1]
                # Normalize the list so that all n most visited regions have the same id.
                visited_regions[:] = [item if item not in top_n else -2 for item in visited_regions]
                # Compute the binary entropy.
                if not visited_regions:
                        continue
                user_binary_entropy = binary_entropy(len(visited_regions), visited_regions.count(-2))
                entropies[i] = user_binary_entropy
                p = visited_regions.count(-2) / len(visited_regions)
                user_prediction_limit = compute_pi_fano(user_binary_entropy, visited_regions)
                limits[i] = user_prediction_limit
        return entropies, limits

def lzw_encode(symbols):
        encoded_symbols = []
        sequence_sizes = []
        codes = defaultdict(int)
        for i, s in enumerate(set(symbols)):
                codes[s] = i
        next_code = max(codes.values()) + 1
        current_string = ""
        for s in symbols:
                current_string = current_string + s
                if not current_string in codes:
                        codes[current_string] = next_code
                        next_code = int(next_code) + 1
                        current_string = current_string[:-len(s)]
                        encoded_symbols.append(codes[current_string])
                        sequence_sizes.append(len(current_string))
                        current_string = str(s)
        encoded_symbols.append(codes[current_string])
        return encoded_symbols, sequence_sizes

def compute_lzw_entropies(df, region_column_name):
        entropies = defaultdict(float)
        limits = defaultdict(float)
        for i, user_checkins in df.groupby(['user_id']):
                visited_regions = list(user_checkins[region_column_name])
                # Remove unkown regions form the list.
                visited_regions[:] = [item for item in visited_regions if item != -1]
                if not visited_regions:
                        continue
                visited_regions = [str(r) for r in visited_regions]
                symbols, ssizes = lzw_encode(visited_regions)
                if not ssizes:
                        continue
                avg_seq_size = sum([len(r) for r in visited_regions]) / len(visited_regions)
                seq_sizes = [(s / avg_seq_size) for s in ssizes]
                real_entropy = (1 / (sum(seq_sizes) / len(seq_sizes))) * math.log(len(seq_sizes))
                entropies[i] = real_entropy
                user_prediction_limit = compute_pi_fano(real_entropy, len(symbols))
                limits[i] = user_prediction_limit
        return entropies, limits
