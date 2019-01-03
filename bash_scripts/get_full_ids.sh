cd /warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/jq-1.5
sosweet_ids=/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/UKSOC_rep/full_sosweet_data/sosweet_ids.txt
finalsosweet_ids=/warehouse/COMPLEXNET/jlevyabi/TWITTERSES/ml_soc_econ/data_files/UKSOC_rep/full_sosweet_data/final_sosweet_ids.txt

rm $sosweet_ids
rm $finalsosweet_ids

for f in /warehouse/COMPLEXNET/TWITTER/data/2014*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -r '.twitter.user.id'|sort -u >> $sosweet_ids
done

sort -u $sosweet_ids >>$finalsosweet_ids
sort -u $finalsosweet_ids >$sosweet_ids
mv $sosweet_ids  $finalsosweet_ids

for f in /warehouse/COMPLEXNET/TWITTER/data/2015*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -r '.twitter.user.id'|sort -u>> $sosweet_ids
done

sort -u $sosweet_ids >>$finalsosweet_ids
sort -u $finalsosweet_ids >$sosweet_ids
mv $sosweet_ids  $finalsosweet_ids

for f in /warehouse/COMPLEXNET/TWITTER/data/2016*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -r '.actor.id'|sort -u>> $sosweet_ids
done

sort -u $sosweet_ids >>$finalsosweet_ids
sort -u $finalsosweet_ids >$sosweet_ids
mv $sosweet_ids  $finalsosweet_ids

for f in /warehouse/COMPLEXNET/TWITTER/data/2017*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -r '.actor.id'|sort -u>> $sosweet_ids
done

sort -u $sosweet_ids >>$finalsosweet_ids
sort -u $finalsosweet_ids >$sosweet_ids
mv $sosweet_ids  $finalsosweet_ids

for f in /warehouse/COMPLEXNET/TWITTER/data/2018*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|./jq -r '.actor.id'|sort -u>> $sosweet_ids
done

sort -u $sosweet_ids >>$finalsosweet_ids
sort -u $finalsosweet_ids >$sosweet_ids
mv $sosweet_ids  $finalsosweet_ids
