#!/bin/bash

#FastInfantSurfer_dHCP_to_mcrib_LUT.tsv # FastInfantSurfer_dHCP_to_mcrib_LUT.tsv  #iBEAT_LUT.tsv #
#csv=/groups/ag-reuter/datasets/FETA/docs_and_meta/subject_list_full.csv
#csv=/groups/ag-reuter/projects/datasets/MCRIB/subjects_mcrib.csv

labels="_full"
base=/groups/ag-reuter/projects
lut=${base}/NeonateVINNA/experiments/LUTs/FastInfantSurfer_dHCP${labels}_LUT.tsv
csv=${base}/NeonateVINNA/Dataset_splits/dataset_split_large_validation_t1t2.csv #iseg_dirs.csv   #uc_davis_dirs.csv #dataset_split_large_testing_t1t2.csv

gpu=0
mode=T2
augS="_RotTLScalePlus" # "_LatentAug" # "_LatentAugRotTLScale" # "_RotTLScale"
synthseg="" # "SynthSegMix" "SynthSegFull"
setsuffix="ValidationSet" # "TestingSet"
net="FastSurferDDB"  #"FastSurferVINN" # FastSurferSVINN
view="all"
suff="AffineNN"
processing="--save_img" #"--load_pred_from_disk --metrics" # enable to get metrics
#

function usage()
{
cat << EOF

Usage:
1. go to gpu cluster
2. start program with arguments (chmod +x ./training_runner.sh to make it executable):
 nohup ./training_runner.sh --net <network (FastSurferVINN, FastSurferDDV, FastSurferSVINN)> \
                            --augS <augmentation (_RotTlScale, _LatentAug, _LatentAugRotTLScale)> \
                            --mode <modality (T1, T2)> \
                            --view <anatomical plane (coronal, axial, sagittal, all (default))> \
                            --csv <csv-file with subjects to analyse> \
                            --gpu <gpu id (0-6)> \
                            --base <base directory with code (/projects)> \
                            --processing <processing to run "--save_img" or "--load_pred_from_disk --metrics>" \
                            --synthseg <optional synthseg processing ("", SynthSegMix, SynthSegFull)> \

training_runner.sh takes input arguments to change training (mode and network). Runs for
three network main networks are supported:
     (i)  FastSurferVINN - voxel size independent neural network (T1 or T2, latent augmentatio, external augmentation,
                                                                            latent augmentation + external augmentation)
     (ii) FastSurferDDB - CNN model with same architecture as VINN (T1 or T2, external augmentation)
     (iii) FastSurferSVINN - voxel size independent neural network with interpolation to any resolution
                             (no final skip connection = FlexSurfer) (T1 or T2, latent augmentatio, external augmentation,
                                                                            latent augmentation + external augmentation)

FLAGS:
  --net <network for inference>         Which network to use for inference: FastSurferDDB (=CNN),
                                                                            FastSurferVINN (=VINN, default),
                                                                            FastSurferSVINN (=FlexVINN)
  --view <output directory>            Base output directory (equivalent to FreeSurfer SUBJECT_DIRECTORIES)
  --mode <image modality>               Input image modality. Either T2 (default), T1, T1 and T2.
  --base <base directory>               Path to Directory with code (/projects for singularity/docker)
  --augS <augmentation>                 Image augmentation to use. One of _LatentAug (for VINNA; default),
                                        RotTLScale (for CNN + exA or VINN + exA), _LatentAugRotTLScale (for VINNA + exA).
                                        Add Plus to any of them to enable intensity augmentation.
  --gpu <gpu id>                        GPU number to run training on.
  --setsuffix <model name suffix>       Suffix for model name (img save name; default = ValidationSet).
  --csv <csv-file>                      Csv-file with subjects to analyse (dataset_split_large_validation_t1t2.csv,
                                        dataset_split_large_testing_t1t2.csv)
  -h --help                             Print Help

REFERENCES:

If you use this for research publications, please cite:

Henschel L, Kuegler D, Zoellei L*, Reuter M* (*co-last),
Orientation Independence through Latent Augmentation,
Imaging Neuroscience, (2023), arxiv.

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A
 fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building
 Resolution-Independence into Deep Learning Segmentation Methods - A Solution
 for HighRes Brain MRI. NeuroImage 251 (2022), 118933.
 http://dx.doi.org/10.1016/j.neuroimage.2022.118933


EOF
}

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --net)
    net="$2"
    shift # past argument
    shift # past value
    ;;
    --augS)
    augS="$2"
    shift # past argument
    shift # past value
    ;;
    --mode)
    mode="$2"
    infermode="$2"
    shift # past argument
    shift # past value
    ;;
    --view)
    view="$2"
    shift # past argument
    shift # past value
    ;;
    --base)
    base="$2"
    shift # past argument
    shift # past value
    ;;
    --synthseg)
    synthseg="$2"
    shift # past argument
    shift # past value
    ;;
    --gpu)
    gpu=$2
    shift # past argument
    shift # past value
    ;;
    --setsuffix)
    add="$2"
    shift # past argument
    shift # past value
    ;;
    --processing)
    add="$2"
    shift # past argument
    shift # past value
    ;;
    --csv)
    csv=${base}/NeonateVINNA/Dataset_splits/"$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    usage
    exit
    ;;
    *)    # unknown option
    echo ERROR: Flag $key unrecognized.
    exit 1
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters


cd $base
for inferres in "05" "08" "10"; do
    if [ "$net" == "FastSurferVINN" ] || [ "$net" == "FastSurferSVINN" ]; then
        br=0.8BR
    else
        br=1.0BR
    fi

    if [ "$infermode" == "T1" ]; then
        orig="T1w"
    else
        orig="T2w"
    fi

    cfg=${net}_Infant_bigMix${suff}${synthseg}_${mode}${augS}_AdamW_Cos_3x3F_71_${br}${labels}
    mn=${setsuffix}_${net}_${mode}_${inferres}_${infermode}${augS}${synthseg}${labels}
    echo $mn $gpu $augS
    #TestingSet_nnUNet_${mode}_${inferres}_${infermode}${augS}${labels} --> --fs --add_subject

    if [ "$view" == "cor" ]; then
        add="--ckpt_cor ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_coronal/Best_training_state.pkl \
              --cfg_cor ${base}/NeonateVINNA/experiments/config/${cfg}_coronal/config.yaml"
    elif [ "$view" == "sag" ]; then
        add="--ckpt_sag ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_sagittal/Best_training_state.pkl \
              --cfg_sag ${base}/NeonateVINNA/experiments/config/${cfg}_sagittal/config.yaml"
    elif [ "$view" == "ax" ]; then
        add="--ckpt_ax ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_axial/Best_training_state.pkl \
            --cfg_ax ${base}/NeonateVINNA/experiments/config/${cfg}_axial/config.yaml"
    else
        add="--ckpt_cor ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_coronal/Best_training_state.pkl \
            --cfg_cor ${base}/NeonateVINNA/experiments/config/${cfg}_coronal/config.yaml \
              --ckpt_ax ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_axial/Best_training_state.pkl \
              --cfg_ax ${base}/NeonateVINNA/experiments/config/${cfg}_axial/config.yaml \
              --ckpt_sag ${base}/NeonateVINNA/experiments/checkpoints/${cfg}_sagittal/Best_training_state.pkl \
              --cfg_sag ${base}/NeonateVINNA/experiments/config/${cfg}_sagittal/config.yaml"
    fi

    if [ "$labels" == "_full" ]; then
        add="${add} --combine"
        suffix="_desc-drawem88"
    else
        suffix="_mapped26_dseg"
    fi

    if [ "$inferres" == "05" ]; then
        orig="${orig}_min.nii.gz"
        gt="${suffix}_dseg_min.nii.gz"
        add="${add} --outdims 320"
    elif [ "$inferres" == "08" ]; then
        orig="${orig}_08_dhcp.nii.gz"
        gt="${suffix}_dseg_08.nii.gz"
        #gt="${suffix}_all_labels_08.nii.gz"
    else
        orig="${orig}.nii.gz"
        gt="${suffix}_dseg.nii.gz"
    fi

    echo $gt $mn
    echo "docker run --gpus device=${gpu} --name henschell_${mn}_gpu${gpu} -v /home/henschell:/home/henschell \
            -v $base:$base -v $base:/fastsurfer --rm --user 4323:1275 -v /groups/ag-reuter/projects/datasets:/groups/ag-reuter/projects/datasets \
            -v /groups/ag-reuter/datasets:/groups/ag-reuter/datasets \
            --shm-size 8G henschell/super_res_surfer:bash \
            nohup python3 $base/SuperResSurfer/run_validation.py --gt_name $gt --orig_name $orig --csv_file $csv \
                                         --add_subject --model_name $mn $processing \
                                         --lut ${lut} \
                          --batch_size 1 $add"
    docker run --name henschell_${mn}_gpu${gpu} -v /home/henschell:/home/henschell \
            -v $base:$base -v $base:/fastsurfer --rm --user 4323:1275 -v /groups/ag-reuter/projects/datasets:/groups/ag-reuter/projects/datasets \
            -v /groups/ag-reuter/datasets:/groups/ag-reuter/datasets \
            --shm-size 8G henschell/super_res_surfer:bash \
            nohup python3 $base/SuperResSurfer/run_validation.py --gt_name $gt --orig_name $orig --csv_file $csv \
                                         --add_subject --model_name $mn $processing \
                                         --lut ${lut} \
                          --batch_size 1 $add > $base/FastInfantSurfer/logs/${mn}.log &

        if [ $gpu -le 6 ]; then
          let "gpu+=1"
        else
          gpu=0
        fi
    done
done
