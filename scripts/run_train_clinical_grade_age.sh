# !/bin/bash

export CUDA_VISIBLE_DEVICES="3" 
now=$(date "+%Y%m%d")"/"$(date "+%H%M%S")


random_seed="2345" # 51 server, grade

cls_mode="grade"
loss="ce"
metric="multiclass"
num_classes=3

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# custom_pretrained_pth="/mai_nas/BYS/mae/output_dir/mae_ch4_75percent/checkpoint-799.pth"
slice_percentile=75

# model="clinical_swint_small"
model="vit_base_clinical_prompt"
clini_info_style="bert"
embed_trainable="False"
decoder_depth="2"
clini_embed_token="word"
# age_cutoff=45
# info_mode="age"
# model="resnet50_lrp,vit_base_lrp,resnet18,resnet50,resnet152,densenet121,densenet201,efficientnet_b0,efficientnet_b2,efficientnet_b4,efficientnetv2_m,efficientnetv2_l,vit_tiny,vit_small,vit_base,swint_tiny,swint_small,swint_base,vit_large,swint_large"

# --------------------------------------------T-R-A-I-N-I-N-G-------------------------------------------------------------------------------------------------------------------
HYDRA_FULL_ERROR=1 python train_clinical.py \
                    loss=$loss metric=$metric \
                    cls_mode=$cls_mode random_seed=$random_seed  paths=train \
                    paths.time_dir=$now data.slice_percentile=$slice_percentile \
                    model=$model model.num_classes=$num_classes \
                    model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
                    model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
                    # # model.age_cutoff=$age_cutoff \
                    # # model.info_mode=$info_mode \
                    # custom_pretrained_pth=$custom_pretrained_pth

# ----------------------------------------------------------------------------------------------------------------------------------------
# INFERENCE

# Using Best AUC Model ---------------------------------------------------------------------------------------------------------------------------------------

save_dir="./exp/runs/"$cls_mode"/"$now"/"

single_ckpt=$save_dir"train/checkpoint_best_auc.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \


python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \


python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \


# Using Best Loss Model ---------------------------------------------------------------------------------------------------------------------------------------

save_dir="./exp/runs/"$cls_mode"/"$now"/"

single_ckpt=$save_dir"train/checkpoint_best_loss.pth"
eval_time=$(date "+%Y%m%d")"_"$(date "+%H%M%S")

python eval_internal_clinical.py task_name=val ckpt=$single_ckpt paths.save_dir=$save_dir paths=internal_eval paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \


python eval_external_clinical.py task_name=TCGA ckpt=$single_ckpt paths.save_dir=$save_dir paths=TCGA paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \


python eval_external_clinical.py task_name=UCSF ckpt=$single_ckpt paths.save_dir=$save_dir paths=UCSF paths.time_dir=$eval_time \
            metric=$metric \
            cls_mode=$cls_mode \
            data.slice_percentile=$slice_percentile  \
            model=$model \
            model.num_classes=$num_classes \
            model.clini_info_style=$clini_info_style model.embed_trainable=$embed_trainable \
            model.decoder_depth=$decoder_depth model.clini_embed_token=$clini_embed_token \
            # # model.age_cutoff=$age_cutoff \
            # # model.info_mode=$info_mode \

