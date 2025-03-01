#!/bin/bash

# Note that this script uses GNU-style sed as gsed. On Mac OS, you are required to first https://brew.sh/
#    brew install gnu-sed
# on linux, use sed instead of gsed in the command below:
# Using sed instead of gsed since we are using Linux
cat ../Data/twitter-datasets/train_pos_full.txt ../Data/twitter-datasets/train_neg_full.txt | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > vocab_full.txt
