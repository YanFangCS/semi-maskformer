source activate dl
python train_net.py --config-file configs/voc_aug/maskformer_R101_bs16_60k.yaml --num-gpus 4 SOLVER.MAX_ITER 120000 SOLVER.IMS_PER_BATCH 16