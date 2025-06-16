CUDA_VISIBLE_DEVICES=0,1 python -u train_total.py \
--learning_rate 0.000004 \
--batch_size  16 \
--database SJTU  \
--img_length_read 6 \
--data_dir_color ./dataset/SJTU-PCQA_maps/6view \
--data_dir_depth ./dataset/SJTU-PCQA_maps/6view_depth \
--num_epochs 50 \
>> logs/total.log

CUDA_VISIBLE_DEVICES=0,1 python -u train_total.py \
--learning_rate 0.000004 \
--batch_size  16 \
--database LS_PCQA_part  \
--img_length_read 6 \
--data_dir_color ./dataset/LS-PCQA_maps/6view \
--data_dir_depth ./dataset/LS-PCQA_maps/6view_depth \
--num_epochs 50 \
>> logs/total.log

CUDA_VISIBLE_DEVICES=0,1 python -u train_total.py \
--learning_rate 0.000004 \
--batch_size 16 \
--database BASICS \
--img_length_read 6 \
--data_dir_color ./dataset/BASICS_maps/6view \
--data_dir_depth ./dataset/BASICS_maps/6view_depth \
--num_epochs 50 \
>> logs/total.log

CUDA_VISIBLE_DEVICES=0,1 python -u train_total.py \
--learning_rate 0.000004 \
--batch_size 16 \
--database WPC \
--img_length_read 6 \
--data_dir_color ./dataset/WPC_maps/6view \
--data_dir_depth ./dataset/WPC_maps/6view_depth \
--num_epochs 50 \
>> logs/total.log
