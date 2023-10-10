python3 train.py \
--log_dir runs/D13.1.2 \
--exp_name hitnet_indemind \
--model HITNetXL_SF \
--gpus 1 \
--check_val_every_n_epoch 5 \
--max_steps 500000 \
--accelerator ddp \
--max_disp 320 \
--max_disp_val 192 \
--optmizer Adam \
--lr 4e-4 \
--lr_decay 400000 0.25 408000 0.1 410000 0.025 \
--lr_decay_type Lambda \
--batch_size 2 \
--batch_size_val 2 \
--num_workers 2 \
--num_workers_val 2 \
--data_augmentation 1 \
--data_type_train INDEMIND \
--data_root_train /data/DEPTH/file_list/d3.5.9/hitnet/train \
--data_list_train lists/kitti2015_train180.list \
--data_size_train 576 320 \
--data_type_val INDEMIND \
--data_root_val /data/DEPTH/file_list/d3.5.9/hitnet/val \
--data_list_val lists/kitti2015_val20.list \
--data_size_val 576 320 \
--init_loss_k 3
