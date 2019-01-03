#!/usr/bin/env python
#encoding: utf-8
import sys
import pandas as pd
import tweepy #https://github.com/tweepy/tweepy
import csv
from tqdm import tqdm
import re
import pickle

import pickle
data_ses=pickle.load(open("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_dic.p","rb"))
geo_usrs=[str(x) for x in data_ses.keys()]

#Twitter API credentials
consumer_key = "ya6Z9E5H4um3qcKx4VyS4fbWs" #"G6LrnHdGQcW7hZGsSQRT1gnkQ"
consumer_secret = "MDzVDYsCd7FIE6oEWpNhIQOwn8xxNus2ZsGpzIlOP95LVGs6bk" #"ewUYjRO7rZC1l97HZHRy8Pt9AkzfqY2YXRk10xUqWMAnQU47D0"
access_key = "250796042-ruLLMdvAlx5RDCzrhuVXw0uYAbCflE1LF6r4pupF" #"836532209322508289-1xr3ZxA4ObClD0Ixx2a81MwnF2dR4da"
access_secret = "Mbkm7rqVxGO453gltLL6KND1sP3qRpZjbcQ4ewQBMpnAw" #"G9rx8oMPreqRYK6JwLIJ7XlBHz2Ute1Jc54YrOm4oI6Rs"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

geo_profiles=[]
deleted_accounts=[]

for usr in tqdm(geo_usrs):
	try:
		geo_profiles.append(api.lookup_users(user_ids=[usr,]))
	except tweepy.TweepError as e:
		if e.api_code==17:
			print("User %s not found in twitter"%usr)
			deleted_accounts.append(usr)
		elif e.api_code==88:
			print("Maximal requests ")			
		else:
			print(e)

pickle.dump(geo_profiles,open("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles.p","wb"))
pickle.dump(deleted_accounts,open("/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/all_together_profiles_deleted_accounts.p","wb"))

