cd /datastore/complexnet/jlevyabi/ml_soc_econ/jq-1.5
g1=/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/2014-2015_locs.txt
g2=/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/2016-2018_locs.txt
#flocs=/datastore/complexnet/jlevyabi/geoloc/txt_files/fake_locs.txt
#rm $g1
rm $g2
nb_fields=3

for f in /datastore/complexnet/twitter/data/2014*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.interaction.geo.latitude!=null and .twitter.place.place_type!="admin")|[.twitter.user.id,.twitter.created_at,.interaction.geo.latitude,.interaction.geo.longitude,.interaction.content]|@tsv'|awk -F '\t' 'NF>$nb_fields {print}'| sed "s/id:twitter.com://" >> $g1
done

for f in /datastore/complexnet/twitter/data/2015*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|select(.interaction.geo.latitude!=null and .twitter.place.place_type!="admin")|[.twitter.user.id,.twitter.created_at,.interaction.geo.latitude,.interaction.geo.longitude,.interaction.content]|@tsv'|awk  -F '\t' 'NF>$nb_fields {print}'| sed "s/id:twitter.com://" >>$g1
done

for f in /datastore/complexnet/twitter/data/2016*.tgz
do
       echo $f
       zcat $f | sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.object.summary]|@tsv'|awk  -F '\t' ' ($3!=""){print }'| sed "s/id:twitter.com://" >>$g2

done

for f in /datastore/complexnet/twitter/data/2017*.tgz
do
       echo $f
       zcat $f | sed "s/^[^{]*//g"|./jq -R 'fromjson?' |./jq -r '.|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.object.summary]|@tsv'|awk  -F '\t' ' ($3!=""){print }'| sed "s/id:twitter.com://" >>$g2
done

for f in /datastore/complexnet/twitter/data/2018*.tgz
do
       echo $f
       zcat $f | sed "s/^[^{]*//g"|jq -R 'fromjson?' |./jq -r '.|[.actor.id,.postedTime,.geo.coordinates[0],.geo.coordinates[1],.object.summary]|@tsv'|awk  -F '\t' ' ($3!=""){print }'| sed "s/id:twitter.com://" >>$g2
done


# quitar falsas locs
#cat $g| awk -F '\t' '{print $3" "$4}'| sort -k1,1 -k2,2| uniq -c | awk '{if($1>50) print $2"-"$3}'> $flocs
#cat $g| awk -F '\t' '{print $1"\t"$2"\t"$3"-"$4}'> temp
#paste -sd '|' $flocs | xargs -I{} grep -v -E {} temp >/datastore/complexnet/jlevyabi/geoloc/txt_files/filtered-2014-2015_locs.txt
#rm temp

