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
model_list=( GCN SpGAT MPNN ChebyNet )
expansion_list=( line_expansion clique_expansion line_graph star_expansion_1 star_expansion_2 lawler_expansion_1 lawler_expansion_2 )

for dname in ${dataset_list[*]}
do
  for model_name in ${model_list[*]}
  do
    for expansion_name in ${expansion_list[*]}
    do
      python graph_pipeline.py $dname $model_list $expansion_list
    done
  done
done
