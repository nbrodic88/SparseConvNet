# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!~/bin/bash
#Download  https://shapenet.cs.stanford.edu/iccv17/ competition data.
#We re-split the Train/Validation data 50-50 to increase the size of the validation set.

wget https://www.dropbox.com/s/9to7yq1nmp5q4mf/Train-Track4.zip?dl=0 -O Train-Track4.zip
wget https://www.dropbox.com/s/8lyklqxiv3uijg0/Train-Track4-Truth.zip?dl=0 -O Train-Track4-Truth.zip
wget https://www.dropbox.com/s/kja8sc464kzxsoy/Test-Track4.zip?dl=0 -O Test-Track4.zip
unzip Train-Track4.zip
unzip Train-Track4-Truth.zip
unzip Test-Track4.zip
for x in train_val test; do 
    for y in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
        mkdir -p $x/$y
    done
done
for x in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
    mv train_*/$x/* val_*/$x/* train_val/$x/; cp test_*/$x/* test/$x/
done
rm -rf train_data train_label val_data val_label test_data test_label
for x in train_val/*/*.pts; do 
    y=`md5sum $x|cut -c 1|tr -d 89abcdef`
    if [ $y ]; then 
        mv $x $x.train
    else 
        mv $x $x.valid
    fi
done
