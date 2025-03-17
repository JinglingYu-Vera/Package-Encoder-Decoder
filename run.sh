python scripts/inference.py \
--exp_dir=/home/tyuah/projects/pixel2style2pixel/experiment_pca_rotation/test \
--checkpoint_path=/home/tyuah/projects/pixel2style2pixel/experiment_new_test/checkpoints/best_model.pt \
--data_path=/home/tyuah/projects/pixel2style2pixel/data/test_pre/test_shampoo/ \
--test_batch_size=7 \
--test_workers=4 \
--couple_outputs \
--resize_outputs