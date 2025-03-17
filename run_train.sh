CUDA_VISIBLE_DEVICES=3 python ./scripts/train.py \
--dataset_type=ffhq_encode \
--exp_dir=/home/tyuah/projects/pixel2style2pixel/experiment_new_test \
--checkpoint_path=/home/tyuah/projects/pixel2style2pixel/experiment_new/checkpoints/best_model.pt \
--workers=8 \
--batch_size=2 \
--test_batch_size=2 \
--test_workers=8 \
--val_interval=2500 \
--save_interval=10000 \
--encoder_type=GradualStyleEncoder \
--start_from_latent_avg \
--lpips_lambda=0.8 \
--l2_lambda=1.5 \
--id_lambda=0.1 \
--output_size=256 \

