# Title

Non-Official PyTorch implementation of []()

# Updates
*   [02/2023] Push Initial rep on Github
*   [--/2023] Release Official code (when the paper is accepted)

# Overview

> abstract

# Getting Started

## Installation
```bash
git clone https://github.com/ewigspace1910/xxxxx
conda create --name XX
conda activate XX
pip install -r requirements.txt
```

## Installation
1. Download the object re-ID datasets Market-1501, Duke-MTMC,... to /datasets. The directory should look like:

```
XX/datas
├── Market-1501-v15.09.15
├── DukeMTMC-reID
|── ....
├── 4Gan #(Must initiated)
└── SystheImgs #(Must initiated)
```

2. To prepare data for GAN, we setup into /datasets/4Gan as following:

```
XX/datas/4Gan
├── duke2market
|   ├──train
|   |   ├──dukemtmc
|   |   ├──market1501c0
|   |   ├──market1501c1
|   |   ├──market1501c2
|   |   ├──market1501c3
|   |   ├──market1501c4
|   |   ├──market1501c5
|   ├──test
|   |   ├──dukemtmc
|   |   ├──market1501c0
|   |   ├──market1501c1
|   |   ├──market1501c2
|   |   ├──market1501c3
|   |   ├──market1501c4
|   |   ├──market1501c5
├── market2duke
|   ├──...
├── ...
```

# Training
We utilize 1 Tesla T4 GPU 16G for training. 
We use 256x128 sized images for Market-1501 and DukeMTMC in both Training-GAN and Training-Reid

- For convenient, we utilize bash script to setup commands. You can reuse or modify them in './XX/scripts'

## Training-GAN
- setup env
```bash
cd XX/CCC/stargan
conda activate XXX
```

- for duke-->market:
```bash
# Train StarGAN on custom datasets
LABEL_DIM=7
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="../../datasets/ReidGan/duke2mark/train"
BATCHSIZE=16
Lidt=1
Lrec=10
Lgp=10
Lcls=1

python main.py --mode train --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-duke2mark/samples \
               --log_dir ../../saves/Gan-duke2mark/logs \
               --model_save_dir ../../saves/Gan-duke2mark/models \
               --result_dir ../../saves/Gan-duke2mark/results \
               --lambda_idt $Lidt \
               --lambda_rec $Lrec \
               --lambda_gp $Lgp --lambda_cls $Lcls
```

- for market-->duke:
```bash
# Train StarGAN on custom datasets
LABEL_DIM=7
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="../../datasets/ReidGan/market2duke/train"
BATCHSIZE=16
Lidt=1
Lrec=10
Lgp=10
Lcls=1

python main.py --mode train --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-mark2duke/samples \
               --log_dir ../../saves/Gan-mark2duke/logs \
               --model_save_dir ../../saves/Gan-mark2duke/models \
               --result_dir ../../saves/Gan-mark2duke/results \
               --lambda_idt $Lidt \
               --lambda_rec $Lrec \
               --lambda_gp $Lgp --lambda_cls $Lcls
```

- After training, we can use trained-GAN models to gen synthetic datasets for reID :

```bash
# 4 duke2market
LABEL_DIM=7
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="../../datasets/4Gan/duke2mark/train/dukemtmc"
FAKEDIR="../../datasets/SyntheImgs/duke2mark" #!!!!
BATCHSIZE=1 #!!!!
ITER=200000
DOMAIN=0

python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-duke2mark/samples \
               --log_dir ../../saves/Gan-duke2mark/logs \
               --model_save_dir ../../saves/Gan-duke2mark/models \
               --result_dir ../../saves/Gan-duke2mark/results \
               --test_iters $ITER --except_domain=$DOMAIN \ #!!!!
               --pattern "{ID}_{CX}_f{RANDOM}.jpg" \ #!!!!
               --gen_dir $FAKEDIR #!!!!
#############################
# market4duke
LABEL_DIM=7
CROP_SIZE=128
IMG_SIZE=128
TRAIN_IMG_DIR="../../datasets/4Gan/mark2duke/train/dukemtmc"
FAKEDIR="../../datasets/SyntheImgs/mark2duke"
BATCHSIZE=1
ITER=200000
DOMAIN=0

python main.py --mode sample --dataset RaFD --rafd_crop_size $CROP_SIZE --image_size $IMG_SIZE \
               --c_dim $LABEL_DIM --rafd_image_dir $TRAIN_IMG_DIR --batch_size $BATCHSIZE\
               --sample_dir ../../saves/Gan-mark2duke/samples \
               --log_dir ../../saves/Gan-mark2duke/logs \
               --model_save_dir ../../saves/Gan-mark2duke/models \
               --result_dir ../../saves/Gan-mark2duke/results \
               --test_iters $ITER --except_domain=$DOMAIN \
               --pattern "{ID}_{CX}_f{RANDOM}.jpg" \
               --gen_dir $FAKEDIR

```

## Training-ReID

### Phase 1: Pretrain
- training on label domain **without** systhetic data(data from GAN)

```bash
#for example
python examples/_source_pretrain.py \
    -ds "dukemtmc" -dt "market1501" \
    -a "resnet50" --feature 0 --iters 200 --print-freq 200\
	--num-instances 4 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
	--logs-dir "../saves/reid/duke2market/S1/woGAN"
    --data-dir "../datasets" \
```


- training on lable domain **with** both real and fake images:
```bash
#for example
python examples/_source_pretrain_fakeimgs.py \
    -ds "dukemtmc" -dt "market1501" \
    -a "resnet50" --feature 0 --iters 400 --print-freq 100\
	--num-instances 4 -b 128 -j 4 --seed 123 --margin 0.3 \
    --warmup-step 10 --lr 0.00035 --milestones 40 70  --epochs 80 --eval-step 1 \
	--logs-dir "../saves/reid/duke2market/S1/wGAN"  \
    --data-dir "../datasets" \
    --fake-data-dir "../datasets/SystheImgs" \
    --lamda 1.
```

* Note: U can modify file 'XXX/CC/modules/datasets/synimgs.py', fragment `ndict` to adapt stucture of Fake Image Folder.

### Phase 2: Finetune
- Modify the below script to excute:
```bash
python examples/_target_finetune_dauet.py \
-dt "market1501" -b 128 -a resnet50part \
--lr 0.00035 --alpha 0.999 --ce-epsilon 0.1 \
--epochs 40 --iters 400 --eval-step 1  --print-freq 100 \
--cluster-eps 0.6  \
--logs-dir "../saves/reid/duke2market/S2/"   \
--data-dir "../datasets" \
--init "../saves/reid/duke2market/S1/wGAN/model_best.pth.tar" \
--ce-weight 1 --tri-weight 1 --soft-ce-weight 0.  --soft-tri-weight 0. \
--aals-epoch 5 --npart 2 --pho 0.2  --uet-al 0.1 --uet-be 0.6  \
--plot \
```

### Evaluate
```bash
python examples/test_model.py \
-dt "market1501" --data-dir "../datasets" \
-a resnet50 --features 0  -b 128 \
--resume ".../model_best.pth.tar" \
--rerank #optional
```

## Acknowledgement


## Citation
If you find this code useful for your research, please consider citing our paper
````BibTex

````


