#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#


dname=$1
method=$2
lr=0.001
wd=0
MLP_hidden=$3
Classifier_hidden=$4
feature_noise=$5
cuda=0

runs=5
epochs=500

dataset_list=( cora citeseer zoo Mushroom house-committees-100 )


#echo ">>>> Model AllDeepSets, Dataset: cora"
#python train.py \
#    --method AllDeepSets \
#    --dname cora \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model AllDeepSets, Dataset: citeseer"
#python train.py \
#    --method AllDeepSets \
#    --dname citeseer \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model AllDeepSets, Dataset: zoo"
#python train.py \
#    --method AllDeepSets \
#    --dname zoo \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001

#echo ">>>> Model AllDeepSets, Dataset: house-committees-100"
#python train.py \
#    --method AllDeepSets \
#    --dname house-committees-100 \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.00001 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.01

#echo ">>>> Model HyperGCN, Dataset: house-committees-100"
#python train.py \
#    --method HyperGCN \
#    --dname house-committees-100 \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.00001 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.01
#
#echo ">>>> Model HyperGCN, Dataset: zoo"
#python train.py \
#    --method HyperGCN \
#    --dname zoo \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 128 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HyperGCN, Dataset: Mushroom"
#python train.py \
#    --method HyperGCN \
#    --dname Mushroom \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HyperGCN, Dataset: cora"
#python train.py \
#    --method HyperGCN \
#    --dname cora \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HyperGCN, Dataset: citeseer"
#python train.py \
#    --method HyperGCN \
#    --dname citeseer \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.00001 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.01
#
#
######
#
#
#
#echo ">>>> Model HCHA, Dataset: house-committees-100"
#python train.py \
#    --method HCHA \
#    --dname house-committees-100 \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.00001 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.01
#
#echo ">>>> Model HCHA, Dataset: zoo"
#python train.py \
#    --method HCHA \
#    --dname zoo \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HCHA, Dataset: Mushroom"
#python train.py \
#    --method HCHA \
#    --dname Mushroom \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HCHA, Dataset: cora"
#python train.py \
#    --method HCHA \
#    --dname cora \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 256 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HCHA, Dataset: citeseer"
#python train.py \
#    --method HCHA \
#    --dname citeseer \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 128 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#
#
#####
#
#
#echo ">>>> Model HGNN, Dataset: house-committees-100"
#python train.py \
#    --method HGNN \
#    --dname house-committees-100 \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HGNN, Dataset: zoo"
#python train.py \
#    --method HGNN \
#    --dname zoo \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HGNN, Dataset: Mushroom"
#python train.py \
#    --method HGNN \
#    --dname Mushroom \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HGNN, Dataset: cora"
#python train.py \
#    --method HGNN \
#    --dname cora \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HGNN, Dataset: citeseer"
#python train.py \
#    --method HGNN \
#    --dname citeseer \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 256 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#
#####
#
#
#echo ">>>> Model HNHN, Dataset: house-committees-100"
#python train.py \
#    --method HNHN \
#    --dname house-committees-100 \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.00001 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.01
#
#echo ">>>> Model HNHN, Dataset: zoo"
#python train.py \
#    --method HNHN \
#    --dname zoo \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 64 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.1
#
#echo ">>>> Model HNHN, Dataset: Mushroom"
#python train.py \
#    --method HNHN \
#    --dname Mushroom \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 128 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HNHN, Dataset: cora"
#python train.py \
#    --method HNHN \
#    --dname cora \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 512 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001
#
#echo ">>>> Model HNHN, Dataset: citeseer"
#python train.py \
#    --method HNHN \
#    --dname citeseer \
#    --All_num_layers 1 \
#    --MLP_num_layers 2 \
#    --Classifier_num_layers 1 \
#    --MLP_hidden 256 \
#    --Classifier_hidden 64 \
#    --wd 0.0000 \
#    --epochs $epochs \
#    --feature_noise 0 \
#    --runs $runs \
#    --cuda $cuda \
#    --lr 0.001

####


echo ">>>> Model CEGCN, Dataset: house-committees-100"
python train.py \
    --method CEGCN \
    --dname house-committees-100 \
    --All_num_layers 1 \
    --MLP_num_layers 2 \
    --Classifier_num_layers 1 \
    --MLP_hidden 512 \
    --Classifier_hidden 64 \
    --wd 0.0000 \
    --epochs $epochs \
    --feature_noise 0 \
    --runs $runs \
    --cuda $cuda \
    --lr 0.001

echo ">>>> Model CEGCN, Dataset: zoo"
python train.py \
    --method CEGCN \
    --dname zoo \
    --All_num_layers 1 \
    --MLP_num_layers 2 \
    --Classifier_num_layers 1 \
    --MLP_hidden 512 \
    --Classifier_hidden 64 \
    --wd 0.0000 \
    --epochs $epochs \
    --feature_noise 0 \
    --runs $runs \
    --cuda $cuda \
    --lr 0.001

echo ">>>> Model CEGCN, Dataset: Mushroom"
python train.py \
    --method CEGCN \
    --dname Mushroom \
    --All_num_layers 1 \
    --MLP_num_layers 2 \
    --Classifier_num_layers 1 \
    --MLP_hidden 64 \
    --Classifier_hidden 64 \
    --wd 0.00001 \
    --epochs $epochs \
    --feature_noise 0 \
    --runs $runs \
    --cuda $cuda \
    --lr 0.01

echo ">>>> Model CEGCN, Dataset: cora"
python train.py \
    --method CEGCN \
    --dname cora \
    --All_num_layers 1 \
    --MLP_num_layers 2 \
    --Classifier_num_layers 1 \
    --MLP_hidden 512 \
    --Classifier_hidden 64 \
    --wd 0.0000 \
    --epochs $epochs \
    --feature_noise 0 \
    --runs $runs \
    --cuda $cuda \
    --lr 0.001

echo ">>>> Model CEGCN, Dataset: citeseer"
python train.py \
    --method CEGCN \
    --dname citeseer \
    --All_num_layers 1 \
    --MLP_num_layers 2 \
    --Classifier_num_layers 1 \
    --MLP_hidden 128 \
    --Classifier_hidden 64 \
    --wd 0.0000 \
    --epochs $epochs \
    --feature_noise 0 \
    --runs $runs \
    --cuda $cuda \
    --lr 0.001