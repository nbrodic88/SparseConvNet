wget https://www.dropbox.com/s/9to7yq1nmp5q4mf/train-val.zip?dl=0 -O train-val.zip
wget https://www.dropbox.com/s/9to7yq1nmp5q4mf/test.zip?dl=0 -O test.zip
unzip train-val.zip
unzip test.zip
#for x in train_val test; do 
#for y in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
#        mkdir -p $x/$y
#    done
#done
#for x in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
#    mv train_*/$x/* val_*/$x/* train_val/$x/; cp test_*/$x/* test/$x/
#done
#rm -rf train_data train_label val_data val_label test_data test_label
#for x in train_val/*/*.pts; do 
#    y=`md5sum $x|cut -c 1|tr -d 89abcdef`
#    if [ $y ]; then 
#        mv $x $x.train
#    else 
#        mv $x $x.valid
#    fi
#done
