#!/bin/bash
#${model}_Infant${sub}${suffix}_AdamW_LROP_3x3F_71_$plane
gpu=$1
plane=$2
classes=$3 
mode=$4
#res=$5
model=FastSurferVINN #FastSurferCNN #FastSurferDDB FastSurferVINN SynthSeg RCVNet
mmodel=FastSurferVINN
net=FastSurferVINN
aug="--aug None --aug Gaussian" #"--aug Scaling --aug Rotation --aug Translation"
laff=True
sub="_big_dHCP" #_MultiModal # _NoScale "" _RotTLScale_Test
modal="" #"--t1_and_t2" # "--t2_only" "--t1_and_t2"

suffix="_T2" #"_ImageAug" #"_T1" #"" "_T1" _RotTLScale '', _T2, _LatentAug, _T2_LatentAug, _RotTLScale, _T2_RotTLScale,
origname=T2w_orig.mgz #T2w_orig.mgz #orig.mgz #,T2w_orig.mgz,mprage.nii.gz,T2w.nii.gz
gtname=dhcp_mapped23_conformed_nn.mgz #mapped23_dseg_fix_05mm.nii.gz #, dhcp_mapped23_conformed_nn.mgz,_mapped23_dseg_fix.nii.gz, _mapped23_dseg_fix.nii.gz
preproc="" # "--add_subject --scale_only" #"--add_subject --scale_only" #"" #, "", "--add_subject --scale_only", "--add_subject --scale_only"
logdir=/autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments
csv=/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/dataset_split_large_validation_t1t2.csv
process="--save_img" #"--metrics --load_pred_from_disk" # "--save_img" #"--metrics --load_pred_from_disk"  "--metrics --save_img"
res="" #"_hires_${res}" #
outdim="256"
#origname=orig_${res}.mgz #T2w_orig.mgz #orig.mgz #,T2w_orig.mgz,mprage.nii.gz,T2w.nii.gz
#gtname=mapped23_dseg_fix_${res}.nii.gz #dhcp_mapped23_conformed_nn.mgz #, dhcp_mapped23_conformed_nn.mgz,_mapped23_dseg_fix.nii.gz, _mapped23_dseg_fix.nii.gz
save_name=ValSet_${model}${sub}${suffix}${res}


if [ "$mode" = "train" ]; then
  echo "$mode == train"
  CUDA_VISIBLE_DEVICES=$gpu python3 /autofs/vast/lzgroup/Users/LeonieHenschel/SuperResSurfer/run_model.py \
           --cfg /autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/${model}_net1.yaml \
           $aug --opt aseg --opt children \
           MODEL.NUM_FILTERS 64 MODEL.KERNEL_H 3 MODEL.KERNEL_W 3 DATA.PLANE $plane \
           OPTIMIZER.LR_SCHEDULER cosineWarmRestarts OPTIMIZER.OPTIMIZING_METHOD adamW \
           TRAIN.NUM_EPOCHS 200 MODEL.NUM_CLASSES $classes TRAIN.BATCH_SIZE 16 DATA.LATENT_AFFINE $laff \
           MODEL.MODEL_NAME $net DATA.PADDED_SIZE 256 MODEL.BASE_RES 1.0 TRAIN.CHECKPOINT_PERIOD 10 \
           MODEL.OUT_TENSOR_WIDTH 256 MODEL.OUT_TENSOR_HEIGHT 256 MODEL.HEIGHT 256 MODEL.WIDTH 256 \
           DATA.PATH_HDF5_TRAIN /autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/training${sub}_$plane.hdf5 \
           DATA.PATH_HDF5_VAL /autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/validation${sub}_$plane.hdf5 \
           LOG_DIR /autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments \
           DATA.LUT /autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/FastInfantSurfer_dHCP_LUT.tsv \
           EXPR_NUM ${model}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_$plane \
           TRAIN.RESUME True TRAIN.RESUME_EXPR_NUM ${model}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_$plane

elif [ "$mode" = "rcvtrain" ]; then
  echo "$mode == rcvtrain"
  CUDA_VISIBLE_DEVICES=$gpu python3 /autofs/vast/lzgroup/Users/LeonieHenschel/SuperResSurfer/run_model.py \
           --cfg /autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/${model}_net1.yaml \
           $aug --opt aseg --opt children \
           MODEL.NUM_FILTERS 64 MODEL.KERNEL_H 3 MODEL.KERNEL_W 3 MODEL.KERNEL_D 3 DATA.PLANE $plane DATA.DIMENSION 3 \
           OPTIMIZER.LR_SCHEDULER cosineWarmRestarts OPTIMIZER.OPTIMIZING_METHOD adamW \
           OPTIMIZER.PATIENCE 50 OPTIMIZER.MODE max OPTIMIZER.FACTOR 0.8 OPTIMIZER.COOLDOWN 25 \
           TRAIN.NUM_EPOCHS 16000 MODEL.NUM_CLASSES $classes TRAIN.BATCH_SIZE 1 DATA.LATENT_AFFINE $laff \
           MODEL.MODEL_NAME $net DATA.PADDED_SIZE 256 MODEL.BASE_RES 1.0 TRAIN.CHECKPOINT_PERIOD 2000 \
           MODEL.OUT_TENSOR_WIDTH 256 MODEL.OUT_TENSOR_HEIGHT 256 MODEL.HEIGHT 256 MODEL.WIDTH 256 \
           DATA.PATH_HDF5_TRAIN /autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/training${sub}_$plane.hdf5 \
           DATA.PATH_HDF5_VAL /autofs/vast/lzgroup/Projects/FastInfantSurfer/hdf5_sets/validation${sub}_$plane.hdf5 \
           LOG_DIR /autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments \
           DATA.LUT /autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/FastInfantSurfer_dHCP_LUT.tsv \
           EXPR_NUM ${model}_Infant${sub}${suffix}_AdamW_3x3F_71_$plane

elif [ "$mode" = "rcv" ]; then
  echo "$mode == RCV eval"
  CUDA_VISIBLE_DEVICES=$gpu python3 /autofs/vast/lzgroup/Users/LeonieHenschel/ThreeDNeuroSeg/eval_w_mgz.py \
           --csv_file $csv \
           --o_dir $logdir/.. \
           --eval_type full \
           --in_name $origname --out_name mri/aseg.$save_name.mgz \
           --cuda_device cuda:0 \
           --base_pretrained_path $logdir/${model}_net1/checkpoints/${model}_Infant${sub}${suffix}_AdamW_CosWR_3x3F_71_$plane/Best_training_state.pkl
else
  echo "$mode == Validation"
  python3 /autofs/vast/lzgroup/Users/LeonieHenschel/SuperResSurfer/run_validation.py \
                          --gt_name $gtname \
                          --orig_name $origname \
                          --model_name $save_name \
                          $preproc $process $modal \
                          --lut /autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/configs/FastInfantSurfer${sub}_LUT.tsv \
                          --csv_file $csv \
                          --out_dir $logdir/${model}_net1/eval_metrics \
                          --ckpt_cor $logdir/${mmodel}_net1/checkpoints/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_coronal/Best_training_state.pkl \
                          --cfg_cor $logdir/${mmodel}_net1/config/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_coronal/config.yaml \
                          --ckpt_ax $logdir/${mmodel}_net1/checkpoints/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_axial/Best_training_state.pkl \
                          --cfg_ax $logdir/${mmodel}_net1/config/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_axial/config.yaml \
                          --ckpt_sag $logdir/${mmodel}_net1/checkpoints/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_sagittal/Best_training_state.pkl \
                          --cfg_sag $logdir/${mmodel}_net1/config/${mmodel}_Infant${sub}${suffix}_AdamW_Cos_3x3F_71_sagittal/config.yaml \
                          --batch_size 4 --outdims $outdim
fi

while read p; do
   a=(${p})
   sid=${a[1]%_*}
   sesid=${a[1]#*_}
   if ! [ -f "/groups/ag-reuter/projects/datasets/dHCP/dhcp_structural_pipeline/onemm/derivatives/${sid}/ses-${sesid}/anat/sub-${sid}_ses-${sesid}_drawem_all_labels.nii.gz" ]; then
        echo "${sid}_${sesid}"
   fi
done < /groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/data/dataset_split_large_training_t1t2_meta.tsv