cd /warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/jq-1.5
profs_14=/warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/icdm18/issues/icdm_geousers_profile_14.txt

#flocs=/warehouse/COMPLEXNET/jlevyabi/geoloc/txt_files/fake_locs.txt
rm $profs_14
cd /warehouse/COMPLEXNET/jlevyabi/ml_soc_econ/jq-1.5

for f in /warehouse/COMPLEXNET/TWITTER/data/2014*.tgz
do
	echo $f
        zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.interaction.geo.latitude!=null and .twitter.place.place_type!="admin")|[.twitter.user.id,.twitter.created_at,.interaction.geo.latitude,.interaction.geo.longitude,.twitter.place.name,.interaction.source,.twitter.user.description,.twitter.user.followers_count,.twitter.user.friends_count,.interaction.content,(.twitter.display_urls|length)]|@tsv'|awk  'NF>9 {print}'| sed "s/id:twitter.com://" >> $profs_14
done


for f in /warehouse/COMPLEXNET/TWITTER/data/2015*.tgz
do
	echo $f
        zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.interaction.geo.latitude!=null and .twitter.place.place_type!="admin")|[.twitter.user.id,.twitter.created_at,.interaction.geo.latitude,.interaction.geo.longitude,.twitter.place.name,.interaction.source,.twitter.user.description,.twitter.user.followers_count,.twitter.user.friends_count,.interaction.content,(.twitter.display_urls|length)]|@tsv'|awk  'NF>9 {print}'| sed "s/id:twitter.com://" >> $profs_14
done

