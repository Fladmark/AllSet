#! /bin/sh
#
# Copyright (C) 2021 
#
# Distributed under terms of the MIT license.
#



dataset_list=( cora citeseer zoo house-committees-100 )
model_list=( MPNN ChebyNet )
expansion_list=( line_expansion clique_expansion line_graph star_expansion_1 star_expansion_2 lawler_expansion_1 lawler_expansion_2 )

for dname in ${dataset_list[*]}
do
  for model_name in ${model_list[*]}
  do
    for expansion_name in ${expansion_list[*]}
    do
      python graph_pipeline.py $dname $model_list $expansion_name
    done
  done
done
