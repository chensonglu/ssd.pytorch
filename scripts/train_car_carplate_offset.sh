CUDA_VISIBLE_DEVICES=0 python train_offset.py --dataset CAR_CARPLATE_OFFSET --dataset_root /data/VALID/720p/car_carplate_offset/VOC/ --save_folder car_carplate_offset_weights/ --lr 1e-4 --visdom True --batch_size 32