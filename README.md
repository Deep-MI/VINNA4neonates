# NeonateVINNA
Newborn Whole Brain Segmentation with VINNA

## VINNA
Contains all code components for running training 
### run_scripts
Bash scripts for running inference and training

### plotting_utils: Final visualizations. 
- Interactive_Plotting: Concatenation of single evaluation measures, enrichment with meta-info, and plotting of simple barplots
- Pooling-illustrations: Code for generation of pooling illustration for the paper
- Seaborn_plottting: Code for generation of final figures for paper
- SuperResTest:
- Inference_Test:
- FinalEval_SetUpTest:

### data_processing
Scripts for dataset splitting, hdf5-generation, data loader, mapping between methods, conversion and affine operations
- conversion:  nifti from dicom and affine transform tests
- data_loader:
	- dataset.py: batch loading routine for reading hdf5-data and passing to model
	- generate_data_hdf5.py: hdf5-set generation
- data_selection:
	- subject_selection.py: randomly split data into training, validation, testing parts.
	- csv_sort_and_split: assign resolutions, add meta information
- mapping:
	- hcp_mapping_script.py: mapping hcp to ibeat
	- dHCP_nnUNet_prep: create nnUNet conform directories and files from dHCP images
	- Mapping_ibeat_for_dsc: Map ibeat and infantFS to common label space with dHCP
- utils: 
	- data_utils: basically code from VINN for thick slice generation, loss function weights calculation, etc. Also rotation affine generation stuff 
	
## Dataset_splits
Meta information and splits for training, validation, testing

## Experiments
Contains output of experiments
- checkpoints: Trained checkpoints
- config: Associated config files used for running the model

Contains inputs for experiments
- hdf5_sets: hdf5_sets for Training
- LUTs: look-up tables, class ids are read from here. Can also be used for FreeSurfer plotting

## Run commands
### Create hdf5-datasets
Training, validation set for VINN, CNN + exA, VINNA
valcsv=${base}/Dataset_splits/dataset_split_large_validation_t1t2_meta_hires.tsv
traincsv=${base}/Dataset_splits/dataset_split_large_training_t1t2_meta_hires.tsv

python3 VINNA/data_processing/data_loader/generate_data_hdf5.py \
                                        --hdf5_name ${base}/NeonateVINN/experiments/hdf5_sets/validation_bigMixAffineNN_dHCP_full_${plane}.hdf5 \
                                        --thickness 3 --image_name T1w.nii.gz --t2_name T2w.nii.gz --img_dim 2 --plane $plane \
                                        --lut $lut --processing none --gt_name _desc-drawem88_dseg.nii.gz --add_sbj  \
                                        --csv_file $valcsv --crop_size 256 --data_dir ${base}/datasets/dHCP/Data/ \
                                        --sizes 290 --hires min --hires_w 3 --gm

### Training commands
Log-files will be written into $base/NeonateVINNA/logs

Training CNN + exA
./VINNA/run_scripts/training_runner.sh --net FastSurferDDB --augS _RotTLScale --plane coronal --mode T2 --base /groups/ag-reuter/projects --gpu 0

Training VINN + exA
./VINNA/run_scripts/training_runner.sh --net FastSurferVINN --augS _RotTLScale --plane coronal --mode T2 --base /groups/ag-reuter/projects --gpu 0

Training VINNA
./VINNA/run_scripts/training_runner.sh --net FastSurferVINN --augS _LatentAug --plane coronal --mode T2 --base /groups/ag-reuter/projects --gpu 0 

Training VINNA + exA
./VINNA/run_scripts/training_runner.sh --net FastSurferVINN --augS _LatentAugRotTLScale --plane coronal --mode T2 --base /groups/ag-reuter/projects --gpu 0 

Training nnUNet
docker run --gpus device=$gpu -v /groups/ag-reuter/projects/datasets/dHCP/nnUNet_data:/nnUNet_data \
--name henschell_nnunet_${task}_fold${fold}_gpu${gpu} --shm-size 8G \
-v /groups/ag-reuter/projects/NeonateVINNA/experiments/nnUNet_models:/nnUNet_models \
-e "nnUNet_raw_data_base=/nnUNet_data/nnUNet_raw_data_base" -e "nnUNet_preprocessed=/nnUNet_data/nnUNet_preprocessed" \
-e "RESULTS_FOLDER=/nnUNet_models/nnUNet_trained_models" \
-v /groups/ag-reuter/projects/master-theses/henschell/nnUNet:/nnUNet \
--rm --user 4323:1275 henschell/super_res_surfer:nnunet_bash \
python3 /nnUNet/nnunet/run/run_training.py 2d nnUNetTrainerV2 Task${task}_InfantSegmentation $fold --npz

### Inference commands
- Variable --sd $outputdir sets output directory (pred will be saved under $outputdir/$sbjid/aseg.${model_name}.mgz in this case).
If you leave it out, the output will be stored relative to the ground truth in the validation file under $sbjid/mri/aseg.${model_name}.mgz.
- Processing can also be set to "--load_pred_from_disk --metrics" to load predictions and calculate DSC and ASD. Output
will be stored in $outputdir/eval_metrics/${metric}_${model_name}.tsv if --sd $outputdir is set. 
Otherwise, output will be stored in LOG_DIR from cfg ($base/NeonateVINNA/experiments/eval_metrics)
- Inference runner automatically runs evaluation on 0.5, 0.8 and 1.0. It currently uses gpu 0, gpu 1 and gpu 2. 

Inference CNN + exA
./VINNA/run_scripts/inference_runner.sh --net FastSurferDDB --augS _RotTLScale --mode T2 --view all --processing "--save_img" \ 
                      --csv dataset_split_large_validation_t1t2.csv --setsuffix ValidationSet --sd $outputdir

Inference VINN + exA
./VINNA/run_scripts/inference_runner.sh --net FastSurferVINN --augS _RotTLScale --mode T2 --view all --processing "--save_img" \ 
                      --csv dataset_split_large_validation_t1t2.csv --setsuffix ValidationSet --sd $outputdir

Inference VINNA
./VINNA/run_scripts/inference_runner.sh --net FastSurferVINN --augS _LatentAug --mode T2 --view all --processing "--save_img" \ 
                      --csv dataset_split_large_validation_t1t2.csv --setsuffix ValidationSet --sd $outputdir

Inference VINNA + exA
./VINNA/run_scripts/inference_runner.sh --net FastSurferVINN --augS _LatentAugRotTLScale --mode T2 --view all --processing "--save_img" \ 
                      --csv dataset_split_large_validation_t1t2.csv --setsuffix ValidationSet --sd $outputdir

Inference nnUNet
docker run --gpus device=2 -v /groups/ag-reuter/projects/datasets/dHCP/nnUNet_data:/nnUNet_data \
-v /groups/ag-reuter/projects/NeonateVINNA/experiments/nnUNet_models:/nnUNet_models \
-v /groups/ag-reuter/projects:/fastsurfer \
-e "nnUNet_raw_data_base=/nnUNet_data/nnUNet_raw_data_base" \
-e "nnUNet_preprocessed=/nnUNet_data/nnUNet_preprocessed" \
-e "RESULTS_FOLDER=/nnUNet_models/nnUNet_trained_models" \
-v /groups/ag-reuter/projects/master-theses/henschell/nnUNet:/nnUNet \
--rm --user 4323:1275 henschell/super_res_surfer:nnunet_bash \
python3 nnUNet/nnunet/inference/predict.py \
-i /nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task668_InfantSegmentation/imagesMCRIB \
-o /nnUNet_data/nnUNet_raw_data_base/nnUNet_raw_data/Task668_InfantSegmentation/labelsMCRIB_3D \
-m /nnUNet_models/nnUNet_trained_models/nnUNet/3d_fullres/Task668_InfantSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1 -f 0




