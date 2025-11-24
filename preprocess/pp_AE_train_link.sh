#!/bin/bash
set -euo pipefail
#set -x

id=$1
export CUDA_VISIBLE_DEVICES=$2

bin=/gpfs/flash/home/yr/src/tigon

echo "preprocess..."
$bin/preprocess_h5ad.py 1.add_time/$id.time.h5ad 2.preprocess/$id.pp.h5ad \
	> 2.preprocess/$id.log 2> 2.preprocess/$id.err

echo "tigon_AE..."
$bin/myAE.py 2.preprocess/$id.pp.h5ad 3.tigon_AE/$id.AE.h5ad --dataset $id \
	> 3.tigon_AE/$id.log 2> 3.tigon_AE/$id.err

echo "tigon_train..."
$bin/myTIGON.py --dataset $id --input_h5ad 3.tigon_AE/$id.AE.h5ad --save_dir 4.tigon_train/$id/ --niters 200 \
	> 4.tigon_train/$id.log 2> 4.tigon_train/$id.err

echo "link_cells..."
$bin/link_cells.py --input_h5ad 3.tigon_AE/$id.AE.h5ad --output_h5ad 5.link_cells/$id.link.h5ad --ckpt 4.tigon_train/$id/ckpt.pth \
	> 5.link_cells/$id.log 2> 5.link_cells/$id.err

echo "$id job-done"
