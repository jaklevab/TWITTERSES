cd /datastore/complexnet/jlevyabi/ml_soc_econ/jq-1.5
lkdin_profiles=/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/linkedin/linkedin_usrs_profiles.txt
linkedin_usr_desc=/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/UKSOC_rep/linkedin/linkedin_usrs_summary.txt
rm $lkdin_profiles
rm $linkedin_usr_desc


for f in /datastore/complexnet/twitter/data/2014*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.twitter.user.id,.twitter.user.screen_name,.twitter.user.url,.twitter.links[]?]|@tsv'|sed 's/id:twitter.com://g'| python /datastore/complexnet/jlevyabi/ml_soc_econ/python_scripts/treat_regexp_line.py>> $lkdin_profiles
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.twitter.user.id,.twitter.user.description]|@tsv'|sed 's/id:twitter.com://g'>> $linkedin_usr_desc
done
for f in /datastore/complexnet/twitter/data/2015*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.twitter.user.id,.twitter.user.screen_name,.twitter.user.url,.twitter.links[]?]|@tsv'|sed 's/id:twitter.com://g'| python /datastore/complexnet/jlevyabi/ml_soc_econ/python_scripts/treat_regexp_line.py>> $lkdin_profiles
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.twitter.user.id,.twitter.user.description]|@tsv'|sed 's/id:twitter.com://g'>> $linkedin_usr_desc
done

for f in /datastore/complexnet/twitter/data/2016*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.actor.id,.actor.preferredUsername,.actor.links[]?.href,.gnip.urls[]?.expanded_url]|@tsv'|sed 's/id:twitter.com://g'| python /datastore/complexnet/jlevyabi/ml_soc_econ/python_scripts/treat_regexp_line.py>> $lkdin_profiles
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.actor.id,.actor.summary]|@tsv'|sed 's/id:twitter.com://g'>> $linkedin_usr_desc
done

for f in /datastore/complexnet/twitter/data/2017*.tgz
do
	echo $f
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.actor.id,.actor.preferredUsername,.actor.links[]?.href,.gnip.urls[]?.expanded_url]|@tsv'|sed 's/id:twitter.com://g'| python /datastore/complexnet/jlevyabi/ml_soc_econ/python_scripts/treat_regexp_line.py>> $lkdin_profiles
	zcat $f| sed "s/^[^{]*//g"|grep linkedin.com|./jq -r '.|[.actor.id,.actor.summary]|@tsv'|sed 's/id:twitter.com://g'>> $linkedin_usr_desc
done
