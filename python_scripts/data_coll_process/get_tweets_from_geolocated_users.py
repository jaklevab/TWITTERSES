import argparse
import sys
import pandas as pd
import tweepy #https://github.com/tweepy/tweepy
import csv
from tqdm import tqdm
import re

#Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""
assert (len(consumer_key)>0)&(len(consumer_secret)>0)&(len(access_key)>0)&(len(access_secret)>0), "Please introduce valid API keys for the crawl"

def get_all_tweets(user_id,home_dir):
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
	#transform the tweepy tweets into a 2D array that will populate the csv
	outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
	#write the csv
	with open(home_dir+'%s_tweets.csv'%user_id, 'w') as f:
		writer = csv.writer(f)
		writer.writerow(["id","created_at","text"])
		writer.writerows(outtweets)
	return(True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input")
	parser.add_argument("-h", "--home_dir")
	input_file, home_dir = args.input, args.home_dir
	args = parser.parse_args()
	geo_home_france=pd.read_csv(header=0,index_col=0,filepath_or_buffer=input_file,columns=["usr"])
	geo_usrs=list(geo_home_france.usr)
	for idx in tqdm(geo_usrs):
		success=get_all_tweets(user_id=idx, home_dir=home_dir)
		if not success:
			print(sys.exc_info()[0])
