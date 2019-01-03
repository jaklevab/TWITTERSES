#screen -S w2v_corpus
cd /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/jq-1.5

touch /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt

nb=1000000
for f in /datastore/complexnet/twitter/data/2014*.tgz
do
        echo $f
	zcat $f| sed "s/^[^{]*//g"|head -$nb| ./jq -r '.twitter.text'| sed 's/@[^ ]*/ /g;s/http[^ ]*/ /g'|sed '/RT /d'>>/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt
done

for f in /datastore/complexnet/twitter/data/2015*.tgz
do
        echo $f
        zcat $f| sed "s/^[^{]*//g"|head -$nb| ./jq -r '.twitter.text'| sed 's/@[^ ]*/ /g;s/http[^ ]*/ /g'|sed '/RT /d'>>/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt
done

for f in /datastore/complexnet/twitter/data/2016*.tgz
do
        echo $f
	zcat $f| sed "s/^[^{]*//g"|head -$nb| ./jq -r '.body'| sed 's/@[^ ]*/ /g;s/http[^ ]*/ /g'|sed '/RT /d'>>/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt
done

for f in /datastore/complexnet/twitter/data/2017*.tgz
do
        echo $f
        zcat $f| sed "s/^[^{]*//g"|head -$nb| ./jq -r '.body'|  sed 's/@[^ ]*/ /g;s/http[^ ]*/ /g'|sed '/RT /d'>>/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt
done

shuf /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt > /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus.txt
rm /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/tweet_corpus.txt
cp /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus.txt /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_safe.txt

sed -i "y/âāáǎàêēéěèîīíǐìôōóǒòûūúǔùǖǘǚǜÂĀÁǍÀÊĒÉĚÈÎĪÍǏÌÔŌÓǑÒÛŪÚǓÙǕǗǙǛ/aaaaaeeeeeiiiiiooooouuuuuüüüüaaaaaeeeeeiiiiiooooouuuuuüüüü/" /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus.txt
cat /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus.txt| tr "[:upper:]" "[:lower:]"|sed "s/'/ /g"| tr -d "[:punct:]"| sed 's/"//g'|sed 's/[?!;,*+<>-]/ /g'|sed 's/[0-9]//g'|awk '{if(NF>5)print}'>/datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus2.txt
mv /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus2.txt /datastore/complexnet/jlevyabi/ml_soc_econ/data_files/Completed_Data/rm_sorted_twitter_corpus.txt


screen -S w2v_model
cd /local/jlevyabi/ml_soc_econ/python_scripts
source Librairies/venv_ml_lib_mine/bin/activate
python words2class_w2v.py
deactivate
screen -X -S w2v_model quit
logout
screen -X -S w2v_corpus quit
logout

