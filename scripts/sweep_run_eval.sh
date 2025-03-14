# !/bin/bash

export CUDA_VISIBLE_DEVICES="0" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")


cls_mode="idh"
metric="binary"
num_classes=1

# --------------------------------------------

slice_percentile=75
end_num_of_exp=7 # num of exp - 1

save_dir="./exp/multiruns/idh/20231214/153634/"
model=(resnet18 resnet50 resnet152 densenet121 densenet201 efficientnet_b0 efficientnet_b2 efficientnet_b4)


# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

    echo $save_dir
    echo ${model[i]}
    BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir \
                paths.job_num=$job_num paths.time_dir=$eval_time metric=$metric \
                model.num_classes=$num_classes model=${model[i]} cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

    python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                best_thres=$BEST_THRES

    python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                best_thres=$BEST_THRES
done



# Using Best Loss Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

    echo $save_dir
    echo ${model[i]}
    BEST_THRES=$(python eval_internal.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir \
                paths.job_num=$job_num paths.time_dir=$eval_time metric=$metric \
                model.num_classes=$num_classes model=${model[i]} cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

    python eval_external.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                best_thres=$BEST_THRES

    python eval_external.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                best_thres=$BEST_THRES

done