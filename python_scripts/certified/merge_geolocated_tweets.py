import pandas as pd
import os
from tqdm import tqdm
import re
import unidecode

#  Text Preprocessing of Tweets (Removes emojis and special characters + lowercase tweets)
tweet_clean = lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",unidecode.unidecode(x.replace("_",""))).split())
tweet_lower=lambda x: x.lower()

# Replace column with values obtained from applying function to column
def apply_inplace(df, field, fun):
    return pd.concat([df.drop(field, axis=1), df[field].apply(fun)], axis=1)

# Compose functions operator
def compose2(f, g):
    return lambda x: f(g(x))

# Preprocessing of tweets for each user
def individual_file_treatment(file_to_treat):
    data_to_return=[]
    f=open(file_to_treat,"r")
    data=list(f.readlines())
    for it in range(1,len(data)):
        info=data[it].split(",")
        if len(info)>=3 and info[0].isnumeric():
            tweet_id,tweet_date,tweet_text=info[0],info[1],info[2:]
            data_to_return.append((tweet_id,tweet_date," ".join(tweet_text)))
        else:
            if len(data_to_return)==0:
                print(data[it])
                continue
            inter_tweet_id,inter_tweet_date,inter_tweet_text=data_to_return[-1]
            inter_tweet_text+=" ".join(info[:])
            data_to_return[-1]=inter_tweet_id,inter_tweet_date,inter_tweet_text
    return (pd.DataFrame(data_to_return,columns=["tweet_id","tweet_date","tweet_text"]))


# Checks if path already exists before deleting any previous attempt to produce the output
if os.path.exists(output_file):
    entered=input("Previous file of merged tweets will be removed. Do you want to continue?")
    if entered=="y":
        pass
    else:
        raise ValueError(' Execution stopped to prevent previous merged file removal.')


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("-h", "--home_dir")
   parser.add_argument("-o", "--output_file")
   args = parser.parse_args()
   home_dir,output_file=args.home_dir,args.output_file
   # Applies previous function to individual twitter users crawl + merges result into ouput file
   for it,file in tqdm(enumerate(os.listdir(home_dir))):
       if file.endswith(".csv"):
           file_name=os.path.join(home_dir, file)
           data_pd=individual_file_treatment(file_name)
           user_id=file.split("_")[0]
           #Retweet removal
           no_rt_data_pd=data_pd[~data_pd.tweet_text.str.contains("RT ")]
           #Rem The following regex just strips of an URL (not just http, any punctuations, User Names or Any non alphanumeric characters + Lowercasing + Accent Stripping
           no_rt_data_pd=apply_inplace(no_rt_data_pd, "tweet_text",compose2(tweet_lower,tweet_clean))
           no_rt_data_pd["user_id"]=[user_id for _ in range(no_rt_data_pd.shape[0])]
           no_rt_data_pd=no_rt_data_pd[["user_id","tweet_id","tweet_date","tweet_text"]]
           if it==0:
               no_rt_data_pd.to_csv(output_file, header=True,index=False,sep=";")
           with open(output_file, 'a') as f:
               no_rt_data_pd.to_csv(f, header=False,index=False,sep=";")
