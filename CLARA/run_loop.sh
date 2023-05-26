#!/bin/sh
r=0.6
for bsz in 64;
do
	for alp in 0.99;
	do
		for lnum in 5;
		do
#	    python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size $bsz --alpha $alp --num-layer $lnum --r ${r} &&
	    wait
#		  python generate.py --gpu 0 --dataset 'mydataset' --load-path "20fs_Pretrain_gin_layer_${lnum}_bsz_${bsz}_nce_k_16384_momentum_${alp}_r${r}/current.pth" &&
	    wait
	    python clara/tasks/graph_classification.py --dataset "mydataset" --hidden-size 2 --model from_numpy_graph --emb-path "./20fs_Pretrain_gin_layer_${lnum}_bsz_${bsz}_nce_k_16384_momentum_${alp}_r${r}/mydataset.npy";
		done
	done
done
