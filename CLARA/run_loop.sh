#!/bin/sh
r=0.15
for bsz in 64, 32;
do
	for alp in 0.99, 0.999;
	do
		for lnum in 5;
		do
	    python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size $bsz --alpha $alp --num-layer $lnum --r ${r} &&
	    echo '1';
		  python generate.py --gpu 0 --dataset 'mydataset' --load-path "./Pretrain_moco_True_mydataset_gin_layer_${lnum}_lr_0.005_decay_1e-05_bsz_${bsz}_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_${alp}_r${r}/current.pth" &&
	    echo '2';
	    python clara/tasks/graph_classification.py --dataset "mydataset" --hidden-size 2 --model from_numpy_graph --emb-path "Pretrain_moco_True_mydataset_gin_layer_${lnum}_lr_0.005_decay_1e-05_bsz_${bsz}_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_${alp}_r${r}/mydataset.npy";
		  echo '3';
		done	
	done
done

