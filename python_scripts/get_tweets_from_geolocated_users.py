#!/usr/bin/env python
#encoding: utf-8

import sys
import pandas as pd
geo_home_france=pd.read_csv(header=0,index_col=0,filepath_or_buffer="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/dgeo_france_most_freq.csv")

geo_usrs=list(geo_home_france.usr)

#Get tweets from geousers active enogh
import tweepy #https://github.com/tweepy/tweepy
import csv
from tqdm import tqdm
import re
#Twitter API credentials
consumer_key = "G6LrnHdGQcW7hZGsSQRT1gnkQ"
consumer_secret = "ewUYjRO7rZC1l97HZHRy8Pt9AkzfqY2YXRk10xUqWMAnQU47D0"
access_key = "836532209322508289-1xr3ZxA4ObClD0Ixx2a81MwnF2dR4da"
access_secret = "G9rx8oMPreqRYK6JwLIJ7XlBHz2Ute1Jc54YrOm4oI6Rs"

def get_all_tweets(user_id):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	new_tweets = api.user_timeline(user_id = user_id,count=200)
	#save most recent tweets
	alltweets.extend(new_tweets)
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1;print ("getting tweets before %s" % (oldest))
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		#all subsiquent requests use the max_id param to prevent duplicates
		new_tweets = api.user_timeline(user_id = user_id,count=200,max_id=oldest)
		#save most recent tweets
		alltweets.extend(new_tweets)
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
	#cleaned_text = [re.sub(r'http[s]?:\/\/.*[\W]*', '', i.text, flags=re.MULTILINE) for i in alltweets] # remove urls
	#cleaned_text = [re.sub(r'@[\w]*', '', i, flags=re.MULTILINE) for i in cleaned_text] # remove the @twitter mentions 
	#cleaned_text = [re.sub(r'RT.*','', i, flags=re.MULTILINE) for i in cleaned_text] # delete the retweets
	#transform the tweepy tweets into a 2D array that will populate the csv	
	#outtweets = [[tweet.id_str, tweet.created_at, cleaned_text[idx].encode("utf-8")] for idx,tweet in enumerate(alltweets)]
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
	#write the csv	
	with open('/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/tweets/individual_files/%s_tweets.csv'%user_id, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)
	return(True)

if __name__ == '__main__':
	for idx in tqdm(geo_usrs):
		success=False
		i=0
		while not success:
			try:
				success=get_all_tweets(idx)
			except:
				i+=1
				print(sys.exc_info()[0])			
				if i>10: 
					success=True 
