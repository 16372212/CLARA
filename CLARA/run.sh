#python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size $1 --num-layer $2 --alpha $3 --r $4;
#
python generate.py --gpu 0 --dataset 'mydataset' --load-path "./Pretrain_moco_True_mydataset_gin_layer_$2_lr_0.005_decay_1e-05_bsz_$1_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_$3_r$4/current.pth";
#
#python clara/tasks/graph_classification.py --dataset "mydataset" --hidden-size 64 --model from_numpy_graph --emb-path "./Pretrain_moco_True_mydataset_gin_layer_$2_lr_0.005_decay_1e-05_bsz_$1_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_$3_r$4/mydataset.npy";



#python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size 32 --alpha 0.99;
#
#python generate.py --gpu 0 --dataset 'mydataset' --load-path './Pretrain_moco_True_mydataset_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.99_r0.15/current.pth';
##
#python clara/tasks/graph_classification.py --dataset "mydataset" --hidden-size 2 --model from_numpy_graph --emb-path "./Pretrain_moco_True_mydataset_gin_layer_5_lr_0.005_decay_1e-05_bsz_32_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.99_r0.15/mydataset.npy";
