#!/bin/bash

declare -a entries=(
	"db/drjohnson"
	"db/playroom"
	"garden"
	"bonsai"
	"kitchen"
	"stump"
	"room"
	"flowers"
	"counter"
	"bicycle"
	"treehill"
	"tandt/train"
	"tandt/truck"
)

for entry in "${entries[@]}"; do
	echo "Working on $entry..."
	mkdir -p ./output/$entry
 	/home/sankeerth/miniconda3/envs/livegs_shangar/bin/python train_compressed_test_codebook.py -s /scratch/sankeerth/$entry --data_device cuda --output_vq ./output/$entry -m ./output/$entry -r 4 --eval
done
