#!/bin/bash

orig=""
sid=""
sd=""
mode="T2"
infermode=${mode}
labels="_full"
base=/projects
lut=${base}/FastInfantSurfer/configs/FastInfantSurfer_dHCP${labels}_LUT.tsv #iBEAT_LUT.tsv #
augS="_LatentAug"
net="FastSurferVINN"  #"FastSurferVINN"
view="all"
suff="AffineNN" #
processing="--save_img --single_img" #"--load_pred_from_disk --metrics" #
add=""
#
function usage()
{
cat << EOF

Usage:
1. go to mlsc cluster
2. start program with arguments (chmod +x ./train_restarter.sh to make it executable):
 sbatch ./single_img_inference.sh --tw <input_image> --sd <output directory> --sid <subject id> --mode <input image modality> [OPTIONS]

single_img_inference.sh takes input arguments to change inference (mode and network). Checkpoints for
two network main networks are supported:
     (i)  FastSurferVINN - voxel size independent neural network (T1 or T2, latent augmentation)
     (ii) FastSurferDDB - CNN model with same architecture as VINN (T1 or T2, external augmentation)

FLAGS:

  --orig <input image>                  Full path to intensity image to run inference on
  --sid <subject id>                    Subject ID, folder created in output directory to save prediction
  --sd <output directory>               Base output directory (equivalent to FreeSurfer SUBJECT_DIRECTORIES)
  --mode <image modality>               Input image modality. Either T2 (default) or T1.
  --base <base directory>               Path to Directory with code (/projects for singularity/docker)
  --lut <look-up-table>                 Look-Up-Table with classes, names and IDs for full and sagittal view
  --augS <augmentation>                 Image augmentation to use. One of _LatentAug (with VINN; default) or _RotTLScale (with CNN)
  --net <network for inference>         Which network to use for inference: FastSurferDDB or FastSurferVINN (default)
  --processing <what to run>            What should be run in addition to inference? Do not neeed to be changed (Default: --single_img --save_img)
  --add <additional arguments>          Additional arguments to add to the call (i.e. --outdims 320 for 0.5 mm images)
  -h --help                             Print Help

REFERENCES:

If you use this for research publications, please cite:

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
    --orig)
    orig="$2"
    shift # past argument
    shift # past value
    ;;
    --sid)
    sid="$2"
    shift # past argument
    shift # past value
    ;;
    --sd)
    sd="$2"
    shift # past argument
    shift # past value
    ;;
    --mode)
    mode="$2"
    infermode="$2"
    shift # past argument
    shift # past value
    ;;
    --base)
    base="$2"
    shift # past argument
    shift # past value
    ;;
    --lut)
    lut="$2"
    shift # past argument
    shift # past value
    ;;
    --augS)
    augS="$2"
    shift # past argument
    shift # past value
    ;;
    --net)
    net="$2"
    shift # past argument
    shift # past value
    ;;
    --processing)
    processing="$2"
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

# Go to base directory (code running started from there)
cd $base

if [ "$net" == "FastSurferVINN" ] || [ "$net" == "FastSurferSVINN" ]; then
    br=0.8BR
    augS="_LatentAug"
else
    br=1.0BR
    augS="_RotTLScale"
fi

cfg=${net}_Infant_bigMix${suff}_${mode}${augS}_AdamW_Cos_3x3F_71_${br}${labels}
mn=FastSurferVINN_LatentAug

add="${add} --ckpt_cor ${base}/FastInfantSurfer/checkpoints/${cfg}_coronal/Best_training_state.pkl \
     --cfg_cor ${base}/FastInfantSurfer/cfgs/${cfg}_coronal/config.yaml \
     --ckpt_ax ${base}/FastInfantSurfer/checkpoints/${cfg}_axial/Best_training_state.pkl \
     --cfg_ax ${base}/FastInfantSurfer/cfgs/${cfg}_axial/config.yaml \
     --ckpt_sag ${base}/FastInfantSurfer/checkpoints/${cfg}_sagittal/Best_training_state.pkl \
     --cfg_sag ${base}/FastInfantSurfer/cfgs/${cfg}_sagittal/config.yaml  \
     --combine --orig_name mri/orig.mgz"

python3 $base/SuperResSurfer/run_validation.py --sd ${sd} --tw ${orig} --sid ${sid}\
                                         --model_name $mn $processing \
                                         --lut ${lut} --batch_size 1 $add