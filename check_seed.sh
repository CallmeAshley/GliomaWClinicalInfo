# !/bin/bash

# 129번 서버에서 아래 코드를 돌렸을 때 아래와 같이 나와야 함
# [2023-12-14 16:18:54,159][train][INFO] - [Epoch 1] [Loss 0.590] [lr 2.0e-05] AUC 0.622
# [2023-12-14 16:19:28,887][val][INFO] - [Epoch 1] [Loss 0.560] AUC 0.672
# [2023-12-14 16:21:07,995][train][INFO] - [Epoch 2] [Loss 0.550] [lr 4.0e-05] AUC 0.702
# [2023-12-14 16:21:41,452][val][INFO] - [Epoch 2] [Loss 0.530] AUC 0.822


export CUDA_VISIBLE_DEVICES="7" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

random_seed="2345" 

cls_mode="grade"
loss="ce"
metric="multiclass"
num_classes=3
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

model="vit_base"

slice_percentile=75
# --------------------------------------------T-R-A-I-N-I-N-G-------------------------------------------------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python train.py \
                    loss=$loss metric=$metric model=$model \
                    cls_mode=$cls_mode random_seed=$random_seed model.num_classes=$num_classes paths=train \
                    paths.time_dir=$now data.slice_percentile=$slice_percentile \
                    # custom_pretrained_pth=$custom_pretrained_pth \
                    # pretrain_which_method=simmim

# --------------------------------------------I-N-F-E-R-E-N-C-E-----------------------------------------------------------------------------------------------------------------
# You should set your sweeping parameter
# model=(resnet18 vit_tiny densenet121)
save_dir="./exp/runs/"$cls_mode"/"$now"/"


single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
job_num=""
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile \
            best_thres=$BEST_THRES

python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
            metric=$metric \
            model.num_classes=$num_classes \
            model=$model \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            best_thres=$BEST_THRES

