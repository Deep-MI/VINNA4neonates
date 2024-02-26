#!/bin/bash

net=FastSurferVINN  #FastSurferSVINN #FastSurferDDB #
augS="_LatentAugRotTLScale" #"_RotTLScalePlus" #"_LatentAug" #ScaleArtefact _RotTLScale _LatentAugUPR
mode=T1 # T1T2 # T2
synthseg="" # "SynthSegMix" # "SynthSegFull" #"SynthSegFull" #
plane=sagittal #sagittal # coronal axial
gpu=0
labels="_full"
suff="AffineNN"

function usage()
{
cat << EOF

Usage:
1. go to gpu cluster
2. start program with arguments (chmod +x ./training_runner.sh to make it executable):
 nohup ./training_runner.sh --net <network (FastSurferVINN, FastSurferDDV, FastSurferSVINN)> \
                            --augS <augmentation (_RotTlScale, _LatentAug, _LatentAugRotTLScale)> \
                            --mode <modality (T1, T2)> \
                            --plane <anatomical plane (coronal, axial, sagittal)> \
                            --labels <full or reduced set (_full)> \
                            --gpu <gpu id (0-6)> \
                            --base <base directory with code (/projects)> \
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
  --plane <anatomical plane>            Anatomical plane to train on (coronal, sagittal, axial)
  --mode <image modality>               Input image modality. Either T2 (default), T1, T1 and T2.
  --base <base directory>               Path to Directory with code (/projects for singularity/docker)
  --augS <augmentation>                 Image augmentation to use. One of _LatentAug (for VINNA; default),
                                        RotTLScale (for CNN + exA or VINN + exA), _LatentAugRotTLScale (for VINNA + exA).
                                        Add Plus to any of them to enable intensity augmentation.
  --gpu <gpu id>                        GPU number to run training on.
  --add <additional arguments>          Additional arguments to add to the call (i.e. --outdims 320 for 0.5 mm images)
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
    --plane)
    plane="$2"
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
    --add)
    add="$2"
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

# Auto determ
nnet=$net #FastSurferVINN

# Set network base res
if [ "$net" == "FastSurferVINN" ] || [ "$net" == "FastSurferSVINN" ]; then
    br=0.8BR
else
    br=1.0BR
fi

# Set Augmentations
if [ "$augS" == "_LatentAug" ]; then
    aug="--aug None --aug Gaussian DATA.LATENT_AFFINE True DATA.UPR_AFFINE False"
elif [ "$augS" == "_LatentAugPlus" ]; then
    aug="--aug RAnisotropy --aug BiasField --aug RGamma --aug Noise --aug Ghosting --aug Blur --aug Gaussian DATA.LATENT_AFFINE True DATA.UPR_AFFINE False"
elif [ "$augS" == "_LatentAugRotTLScale" ]; then
    aug="--aug Scaling --aug Translation --aug Rotation --aug Gaussian DATA.LATENT_AFFINE True DATA.UPR_AFFINE False"
elif [ "$augS" == "_RotTLScale" ]; then
    aug="--aug Scaling --aug Translation --aug Rotation"
elif [ "$augS" == "_RotTLScalePlus" ]; then
    aug="--aug Scaling --aug Translation --aug Rotation --aug RAnisotropy --aug BiasField --aug RGamma --aug Noise --aug Ghosting --aug Blur"
elif [ "$augS" == "" ] && [ "$br" == "0.8BR" ]; then
    aug="--aug None --aug Gaussian DATA.LATENT_AFFINE False DATA.UPR_AFFINE False"
else
    aug="--aug Scaling"
fi

# Set data mode
if [ "$mode" == T1 ]; then
    add="DATA.IMG_TYPE image DATA.MIX_T1_T2 False"
elif [ "$mode" == T2 ]; then
    add="DATA.IMG_TYPE t2_image DATA.MIX_T1_T2 False"
else
    add="DATA.IMG_TYPE t2_image DATA.MIX_T1_T2 True"
fi

# Set epochs. Latent aug needs longer for convergence
if [ "$augS" == "_LatentAug" ] || [ "$augS" == "_LatentAugPlus" ]; then
    cl="TRAIN.NUM_EPOCHS 160"
else
    cl="TRAIN.NUM_EPOCHS 80"
fi

# Set number classes, LUTs
if [ "$plane" == "sagittal" ]; then # 89 full, 47 sagittal
    cl="$cl MODEL.NUM_CLASSES 47"
    add="$add DATA.SYNTHSEG_LUT ${base}/NeonateVINNA/VINNA/config/LUTs/synthseg_infant_test_sagittal.tsv"
else
    cl="$cl MODEL.NUM_CLASSES 89"
    add="$add DATA.SYNTHSEG_LUT ${base}/NeonateVINNA/VINNA/config/LUTs/synthseg_infant_test.tsv"
fi

# Add Synthseg processing if wanted
if [ "${synthseg}" == "SynthSegFull" ]; then
    aug="--aug SynthSeg $aug"
    add="$add DATA.SYNTHSEG_TRADEOFF 0.0"
elif [ "${synthseg}" == "SynthSegMix" ]; then
    aug="--aug SynthSeg $aug"
    add="$add DATA.SYNTHSEG_TRADEOFF 0.5"
fi

# Assemble cfg from inputs
cfg=${net}_Infant_bigMix${suff}${synthseg}_${mode}${augS}_AdamW_Cos_3x3F_71_${br}${labels}_${plane}
cfgOld=${nnet}_Infant_bigMixAffineNN_T2${augS}_AdamW_Cos_3x3F_71_${br}${labels}_coronal
log=$base/NeonateVINNA/logs/training/${net}_${br}_${mode}${labels}${augS}${synthseg}_${plane}.log

# Assemble run call
nohup docker run --gpus device=$gpu \
    --name henschell_${net}_${plane}_${mode}_${suff}${labels}${augS}${synthseg}_gpu${gpu} \
    -v $base:$base --rm --user 4323:1275 \
    --env "PYTHONPATH=/fastsurfer:$base:$base/master-theses/henschell" \
    --shm-size 8G henschell/super_res_surfer:bash \
    python3 $base/NeonateVINNA/VINNA/run_model.py \
        --cfg $base/NeonateVINNA/VINNA/config/${nnet}/${augS:1}/${cfgOld}/config.yaml \
        ${aug} \
        DATA.PATH_HDF5_TRAIN $base/NeonateVINNA/experiments/hdf5_sets/training_bigMix${suff}_dHCP${labels}_${plane}.hdf5 \
        DATA.PATH_HDF5_VAL $base/NeonateVINNA/experiments/hdf5_sets/validation_bigMix${suff}_dHCP${labels}_${plane}.hdf5 \
        DATA.META_INFO $base/NeonateVINNA/Dataset_splits/dataset_split_large_validation_t1t2_meta_slices.tsv \
        DATA.AUGNAME ${augS:1} \
        ${add} ${cl} MODEL.MODEL_NAME ${net} DATA.PLANE ${plane} \
        LOG_DIR $base/NeonateVINNA/experiments \
        SUMMARY_PATH $base/NeonateVINNA/experiments/summary/${nnet}/${augS:1}/ \
        CONFIG_LOG_PATH $base/NeonateVINNA/VINNA/config/${nnet}/${augS:1}/${cfg} \
        DATA.LUT $base/NeonateVINNA/VINNA/config/LUTs/FastInfantSurfer_dHCP${labels}_LUT.tsv \
        EXPR_NUM ${cfg} TRAIN.RESUME False OPTIMIZER.BASE_LR 0.01 TRAIN.BATCH_SIZE 16 > $log &
