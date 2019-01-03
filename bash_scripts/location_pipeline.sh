
## Collection of geolocation information out of the SOSWEET Twitter database
profiles_data="/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/icdm18/issues/icdm_geousers_profile_14.txt"


rm $profiles_data
cd /warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/jq-1.5
echo "Collecting information from users having tweeted with geolocation ..."
echo "Collected information: twitter_id, posting_time, latititude, longitude, city, service, profile description, # followers, # friends, tweet, # urls"

# Locations conatained in 2014-2015 sample
for f in /warehouse/COMPLEXNET/TWITTER/data/201@(4|5)*
do
	echo "Processing file: "$f
        zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' \
|./jq -r '.|select(.interaction.geo.latitude!=null and .twitter.place.place_type!="admin")|[.twitter.user.id,.twitter.created_at,.interaction.geo.latitude,.interaction.geo.longitude,.twitter.place.name,.interaction.source,.twitter.user.description,.twitter.user.followers_count,.twitter.user.friends_count,.interaction.content,(.twitter.display_urls|length)]|@tsv'\
	|awk  'NF>9 {print}'| sed "s/id:twitter.com://" >> $profiles_data
done

tmp_geo_usrs="/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/icdm18/issues/geo_id_usrs.txt"
# Directory where all individual crawls were stored
home_dir="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/tweets/geolocated_users/"
# Output file with all information combined
output_file="/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/tweets/all_geolocated_users.csv"
cat $profiles_data| awk -F '\t' '{print $1}'|sort -n| uniq>  $tmp_geo_usrs


## Collection of last 3,200 tweets for sampled geolocated users
echo "Starting collection of available tweets for the geolocated users"
~/anaconda3/bin/python get_tweets_from_geolocated_users.py -i $tmp_geo_usrs -h $home_dir
echo "Merging Results"
~/anaconda3/bin/python merge_geolocated_tweets.py -h $home_dir -o $output_file

rm $tmp_geo_usrs

