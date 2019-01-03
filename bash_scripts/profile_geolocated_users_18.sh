cd /warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/jq-1.5
profs_18=/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/icdm18/issues/icdm_geousers_profile_18.txt

rm $profs_18
cd /warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/jq-1.5


for f in /warehouse/COMPLEXNET/TWITTER/data/2016*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.geo.coordinates[0]!=null )|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.geo.type,.generator.displayName,.actor.summary,.actor.followersCount,.actor.friendsCount,(.object.twitter_entities.urls|length),.location.name,.location.geo.type,.location.objectType]|@tsv'|awk  -F '\t' 'NF>11 {print}'| sed "s/id:twitter.com://"  >> $profs_18
done

for f in /warehouse/COMPLEXNET/TWITTER/data/2017*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.geo.coordinates[0]!=null )|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.geo.type,.generator.displayName,.actor.summary,.actor.followersCount,.actor.friendsCount,(.object.twitter_entities.urls|length),.location.name,.location.geo.type,.location.objectType]|@tsv'|awk  -F '\t' 'NF>11 {print}'| sed "s/id:twitter.com://"  >> $profs_18
done

for f in /warehouse/COMPLEXNET/TWITTER/data/2018*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.geo.coordinates[0]!=null )|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.geo.type,.generator.displayName,.actor.summary,.actor.followersCount,.actor.friendsCount,(.object.twitter_entities.urls|length),.location.name,.location.geo.type,.location.objectType]|@tsv'|awk  -F '\t' 'NF>11 {print}'| sed "s/id:twitter.com://"  >> $profs_18
done

