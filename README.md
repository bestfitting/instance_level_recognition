# Instance Level Recognition

This repository contains the code for the 2nd place solution to the 2020 edition of the Google Landmark Recognition competition hosted on Kaggle: 

https://www.kaggle.com/c/landmark-recognition-2020/leaderboard

The full solution is described here

https://www.kaggle.com/c/landmark-recognition-2020/discussion/188299

## Definition
v2c(cleaned GLDv2), there are 1.6 million training images and 81k classes. All landmark test images belong to these classes.

v2x,in GLDv2, there are 3.2 million images belong to the 81k classes in v2c. I define these 3.2m images as v2x.

## Data preparation
1.Please config your local directory in CODE_DIR/src/config/config.py

2.Download Google Landmarks Dataset v2 train,test,index from https://github.com/cvdfoundation/google-landmark ,unpack them to DATA_DIR/images

3.Move train.csv,train_clean.csv to DATA_DIR/raw (provided by Kaggle, is not included in my solution file)

4.Download superpoint superglue models from https://github.com/magicleap/SuperPointPretrainedNetwork and https://github.com/magicleap/SuperGluePretrainedNetwork

5.Create split file:
```
python create_split.py
```

## Training retrieval models
### 1. Train EfficientNet B7

1.1 Train EfficientNet B7 v2c 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b7_gem_fc_arcface2_1head --save_every_epoch 0.1 --epochs 7 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2c_sgd_ls_aug1_norm1_0907_class_efficientnet_b7_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2c --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 4 --distributed 1 --preprocessing 1
```

1.2. Train efficientnet_b7 v2x 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b7_gem_fc_arcface2_1head --save_every_epoch 0.1 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2x_sgd_ls_aug3b_norm1_0918_class_efficientnet_b7_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2x --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 4 --distributed 1 --preprocessing 1 --model_file RESULT_DIR/models/v2c_sgd_ls_aug1_norm1_0907_class_efficientnet_b7_gem_fc_arcface2_1head_i448/6.70.pth
```

### 2. Train efficientnet_b6 

2.1 Train efficientnet_b6 v2c 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b6_gem_fc_arcface2_1head --save_every_epoch 0.1 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2c_sgd_ls_aug3b_norm1_0919_class_efficientnet_b6_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2c --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 6 --distributed 1 --preprocessing 1
```

2.2 Train efficientnet_b6 v2x 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b6_gem_fc_arcface2_1head --save_every_epoch 0.1 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2x_sgd_ls_aug3b_norm1_0919_class_efficientnet_b6_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2x --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 6 --distributed 1 --preprocessing 1 --model_file RESULT_DIR/models/v2c_sgd_ls_aug3b_norm1_0919_class_efficientnet_b6_gem_fc_arcface2_1head_i448/6.00.pth
```

### 3 Train efficientnet_b5

3.1 Train efficientnet_b5 v2c 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b5_gem_fc_arcface_1head --save_every_epoch 0.1 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2c_sgd_ls_aug3b_norm1_0918_class_efficientnet_b5_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2c --num_classes 81313 --gpu_id 0,1,2,3 --distributed 1 --preprocessing 1  --batch_size 8
```

3.2 Train efficientnet_b5 v2x 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_efficientnet_b5_gem_fc_arcface_1head --save_every_epoch 0.2 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2x_sgd_ls_aug3b_norm1_0918_class_efficientnet_b5_gem_fc_arcface2_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2x --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 8 --distributed 1 --preprocessing 1 --model_file RESULT_DIR/models/v2c_sgd_ls_aug3b_norm1_0918_class_efficientnet_b5_gem_fc_arcface2_1head_i448/8.20.pth
```

### 4. Train resnet152

4.1 Train resnet152 v2c 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_resnet152_gem_fc_arcface_1head --save_every_epoch 0.2 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2c_sgd_ls_aug3b_norm1_0919_class_resnet152_gem_fc_arcface_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2c --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 10 --distributed 1 --preprocessing 1
```

4.2 Train resnet152 v2x 448x448 model
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --arch class_resnet152_gem_fc_arcface_1head --save_every_epoch 0.2 --epochs 30 --img_size 448 --eval_img_size 512 --scheduler SGD --out_dir v2x_sgd_ls_aug3b_norm1_0919_class_resnet152_gem_fc_arcface_1head_i448 --loss LabelSmoothingLossV1 --aug_version 1 --split_type v2x --num_classes 81313 --gpu_id 0,1,2,3 --batch_size 10 --distributed 1 --preprocessing 1 --model_file RESULT_DIR/models/v2c_sgd_ls_aug3b_norm1_0919_class_resnet152_gem_fc_arcface_1head_i448/7.40.pth
```

## Generate submission
Generate submission is detailed in notebooks/generate_submissions.ipynb.