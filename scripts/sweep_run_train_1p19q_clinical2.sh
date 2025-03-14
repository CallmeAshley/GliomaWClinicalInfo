# !/bin/bash

export CUDA_VISIBLE_DEVICES="1"
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")

# ---------------------------------------
random_seed="3579" # 129 server, idh

cls_mode="1p_19q"
loss="bce"
metric="binary"
num_classes=1

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# custom_pretrained_pth="/mai_nas/BYS/mae/output_dir/mae_ch4_10percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_25percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_50percent/checkpoint-799.pth,/mai_nas/BYS/mae/output_dir/mae_ch4_75percent/checkpoint-799.pth"

model="vit_base_clinical_prompt"

slice_percentile=75
clini_info_style="bert"
embed_trainable="False"
decoder_depth="1,4,8"
clini_embed_token="word"

# --------------------------------------------T-R-A-I-N-I-N-G-------------------------------------------------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python train_clinical.py -m \
                    loss=$loss metric=$metric model=$model \
                    cls_mode=$cls_mode random_seed=$random_seed model.num_classes=$num_classes paths=train \
                    paths.time_dir=$now data.slice_percentile=$slice_percentile \
                    model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                    model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
                    # custom_pretrained_pth=$custom_pretrained_pth \
                    # pretrain_which_method=simmim

# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE
save_dir="./exp/multiruns/"$cls_mode"/"$now"/"

decoder_depth=(1 4 8)
end_num_of_exp=2 # num of exp - 1

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_auc.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
    BEST_THRES=$(python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

    python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile  \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                best_thres=$BEST_THRES

    python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile  \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                best_thres=$BEST_THRES
done


# Using Best Loss Model ---------------------------------------------------------------------------------------------------------------------------------------

for i in $(seq 0 $end_num_of_exp)
do  
    single_ckpt=$save_dir"train/"$i"/checkpoint_best_loss.pth"
    job_num="/"$i
    eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")
    BEST_THRES=$(python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                | tee /dev/tty | grep 'RETURN:' | sed 's/RETURN: //')

    python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile  \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                best_thres=$BEST_THRES

    python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.job_num=$job_num paths.time_dir=$eval_time \
                metric=$metric \
                model.num_classes=$num_classes \
                model=${model[i]} \
                cls_mode=$cls_mode \
                data.slice_percentile=$slice_percentile  \
                model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                model.decoder_depth=${decoder_depth[i]} model.clini_embed_token=$clini_embed_token \
                best_thres=$BEST_THRES
done
