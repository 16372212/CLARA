#python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size 32 --alpha 0.999;

wait

#python generate.py --gpu 0 --dataset 'mydataset' --load-path './50_Pretrain_gin_layer_5_bsz_32_nce_k_16384_momentum_0.999_r0.15/current.pth';

wait

python clara/tasks/graph_classification.py --dataset "mydataset" --hidden-size 2 --model from_numpy_graph --emb-path "./50_Pretrain_gin_layer_5_bsz_32_nce_k_16384_momentum_0.999_r0.15/mydataset.npy";
