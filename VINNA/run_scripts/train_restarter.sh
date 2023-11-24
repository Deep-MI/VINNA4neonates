#!/bin/bash

# Script to restart Training of VINN, sVINN, CNN for T1 or T2 w/wo Augmentation of all planes
# Checks if the training points already exists and starts from latest

# Regular flags defaults
base="/autofs/vast/lzgroup/Users/LeonieHenschel"
logdir=$base
modality="T1"
net="FastSurferVINN"
plane="coronal"
hdf_s="big"
hdf_r="05"
label="dHCP"
ext_aug="--aug None"
int_aug=""
ckpt_aug=""
lat_aug="DATA.LATENT_AFFINE False"
efa_aug="MODEL.ELASTIC_DEFORMATION False"
eladef=""
baseres="0.8"
epochs=200
time_limit="3-12:00:00"
resume=""
data_dim=2
batchsize=16
logfile="log.log"
site=""
gpu="device=0"

function usage()
{
cat << EOF

Usage:
1. go to mlsc cluster
2. activate babySurfer environment ("conda activate babySurfer")
    - yaml-file in ./User/LeonieHenschel/FastInfantSurfer/conda)
    - create environment via "conda env create --file babySurfer.yaml"
3. start program with arguments (chmod +x ./train_restarter.sh to make it executable):
 ./train_restarter.sh --net <network> --modality <img modality> --plane <anatomical plane> [OPTIONS]

 Time limit is set to 3 days, 12 hours and can be extended/shortened by passing the --time_limit flag (d-hours:min:seconds).
 If training should be resumed, pass the --resume flag (automatically chooses correct ckpt based on the other input options)

train_restarter.sh takes input arguments to (re)start training on the mlsc cluster (with jobsubmit) of
the three main networks in combination with different resolutions and augmentations (Rotation, Translation, Scaling):
     (i)  FastSurferVINN - voxel size independent neural network (T1 or T2, latent augmentation, external augmentation, no augmentation)
     (ii) FastSurferSVINN - super resolution VINN (T1 or T2, latent augmentation, external augmentation, no augmentation)
     (iii) FastSurferCNN - standard CNN model (T1 or T2, external augmentation, no augmentation)
     (iv) FastSurferDDB - CNN model with same architecture as VINN (T1 or T2, external augmentation, no augmentation)
     (v) RCVNet - 3D CNN model with same architecture as FastSurferCNN (T1 or T2, no augmentation)

FLAGS:

  --base <base directory>               Path to Directory with code (/autofs/vast/lzgroup/Users/LeonieHenschel)
  --site <site for run call>            Site to adapt run call (MGH: jobsubmit, DZNE: docker, empty (Default): print command
  --plane <anatomical plane>            Plane to start retraining for (coronal (default), axial, sagittal or all for 3D)
  --modality  <modality>                Modality to train on (T1 =default or T2)
  --net <network to train>              Which network to (re)train: FastSurferCNN, FastSurferDDB, FastSurferVINN, FastSurferSVINN, RCVNet
  --labels <segmentation labels>        which labels to use (dHCP = default)
  --hdf_r <hdf5-resolution>             which resolution to use in hdf5-dataset (empty str = 1.0 mm (default), 05 = 0.5 mm, 08 = 0.8 mm)
  --hdf_s <hdf5-size>                   which hdf5-size to use (big=300, empty str "" =30 (default))
  --ext_aug                             Turns on scaling, rotation and translation augmentation
  --int_aug  <str>                      Define intensity augmentations to run (i.e. "--aug BiasField")
  --lat_aug                             Turns on latent space augmentation
  --elastic                             Turn on elastic deformation (in latent space, if --lat_aug, else externally)
  --baseres <base network resolution>   Resolution of the base network inside VINN (0.8 by default, can be 1.0 or 0.5)
  --batchsize <int>                     Batch size for training
  --epochs <int epochs>                 Number of epochs to train (if resume, pick up at last point and train until epoch specified here is reached (default=200)
  --time_limit <"3-12:00:00">           Time limit for jobsubmit. See lcn page for details. Default 3-12:00:00.
  --resume                              Turn on to resume training instead of starting from scratch
  --logfile <log-file name>             Logfilename to store error/stdout in (written in $base/FastInfantSurfer/nohup_logs/$logfile)
  --gpu <str>                           GPU(s) to use for running model (default="device=0"). Use "device=0,2" for multi input.
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
    --base)
    base="$2"
    logdir=$base
    shift # past argument
    shift # past value
    ;;
    --site)
    site="$2"
    shift # past argument
    shift # past value
    ;;
    --gpu)
    gpu="$2"
    shift # past argument
    shift # past value
    ;;
    --plane)
    plane="$2"
    shift # past argument
    shift # past value
    ;;
    --modality)
    modality="$2"
    shift # past argument
    shift # past value
    ;;
    --net)
    net="$2"
    shift # past argument
    shift # past value
    ;;
    --hdf_s)
    hdf_s="$2"
    shift # past argument
    shift # past value
    ;;
    --hdf_r)
    hdf_r="$2"
    shift # past argument
    shift # past value
    ;;
    --baseres)
    baseres="$2"
    shift # past argument
    shift # past value
    ;;
    --batchsize)
    batchsize="$2"
    shift # past argument
    shift # past value
    ;;
    --label)
    label="$2"
    shift # past argument
    shift # past value
    ;;
    --ext_aug)
    ext_aug="--aug Scaling --aug Rotation --aug Translation"
    ckpt_aug="${ckpt_aug}_RotTLScale"
    shift # past value
    ;;
    --int_aug)
    int_aug="$2"
    shift # past argument
    shift # past value
    ;;
    --elastic)
    eladef="_ElasticDeform"
    ckpt_aug="${ckpt_aug}_Elastic"
    shift # past argument
    ;;
    --lat_aug)
    lat_aug="DATA.LATENT_AFFINE True"
    ckpt_aug="${ckpt_aug}_LatentAug"
    shift # past argument
    ;;
    --epochs)
    epochs="$2"
    shift # past argument
    shift # past value
    ;;
    --time_limit)
    time_limit="$2"
    shift # past argument
    shift # past value
    ;;
    --resume)
    resume="TRAIN.RESUME True"
    shift # past argument
    ;;
    --logfile)
    logfile="$2"
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

##
# assign Image type and cfg-file based on input arguments
##
OTW=256
OTH=256
PDS=256

if [ "$net" == "FastSurferCNN" ]; then
      cfg="${base}/FastInfantSurfer/configs/FastSurfer_net1.yaml"
      if ! [ -z ${eladef} ]; then
          ext_aug="${ext_aug} --aug Elastic"
      fi
elif [ "$net" == "FastSurferDDB" ]; then
  cfg="${base}/FastInfantSurfer/configs/FastSurferDDB_net1.yaml"
  if ! [ -z ${eladef} ]; then
     ext_aug="${ext_aug} --aug Elastic"
  fi
elif [ "$net" == "RCVNet" ]; then
  cfg="${base}/FastInfantSurfer/configs/RCVNet_net1.yaml"
  data_dim=3
  plane="all"
else # VINN or SVINN
  cfg="${base}/FastInfantSurfer/configs/FastSurferVINN_net1.yaml"
  ext_aug="${ext_aug} --aug Gaussian"
fi

if ! [ -z ${eladef} ]; then
  if [ "${lat_aug}" == "DATA.LATENT_AFFINE False" ]; then
     # Elastic set, external augmentation only
     ext_aug="${ext_aug} --aug Elastic"
  else
     # Elastic set, latent augmentation
     efa_aug="MODEL.ELASTIC_DEFORMATION True"
  fi
fi

if [ "$modality" == "T1" ]; then
  img_dtype="image"
else # VINN or SVINN
  img_dtype="t2_image"
fi

if ! [ -z ${int_aug} ]; then
  ext_aug="${ext_aug} ${int_aug}"
fi

##
# Construct checkpoint and hdf5-sets from input arguments
##
ckpt="${net}_Infant_${hdf_s}${hdf_r}_${modality}${ckpt_aug}_AdamW_Cos_3x3F_71_${baseres}BR_${plane}"
sub="${hdf_s}${hdf_r}_${label}"

##
# Set up momdel height and width based on base-res
##
if [ "${baseres}" == "0.8" ]; then
  mw=320
  mh=320
elif [ "${baseres}" == "0.5" ]; then
  mw=512
  mh=512
else
  # baseres 1.0
  mw=256
  mh=256
fi

if [ "${plane}" == "all" ]; then
  mw=256
  mh=256
  ckptper=2000
  opt_lr="OPTIMIZER.LR_SCHEDULER reduceLROnPlateau OPTIMIZER.OPTIMIZING_METHOD adamW OPTIMIZER.PATIENCE 50 OPTIMIZER.MODE max OPTIMIZER.FACTOR 0.8 OPTIMIZER.COOLDOWN 25 "
  ckpt="${net}_Infant_${hdf_s}${hdf_r}_${modality}_AdamW_LROP_3x3F_71_${baseres}BR_${plane}"
else
  ckptper=20
  opt_lr="OPTIMIZER.LR_SCHEDULER cosineWarmRestarts OPTIMIZER.OPTIMIZING_METHOD adamW"
fi

# Set up number of classes
num_classes=24
if [ "${plane}" == "sagittal" ]; then
  num_classes=14
fi

if ! [ -z ${resume} ]; then
  resume="TRAIN.RESUME True TRAIN.RESUME_EXPR_NUM ${ckpt}"
fi

logs=${base}/FastInfantSurfer/logs/${logfile}
echo "Retraining ${net} with modality ${modality}, plane ${plane}, hdf5_set ${hdf5_train}, augmentation $ckpt_aug $lat_aug. Saving log-files in $logs"
rm ${logs}

if [ "${site}" == "MGH" ]; then
  callsig="jobsubmit -p rtx8000 -m 10G -t ${time_limit} -o $logs -A zolleigp -M ALL -c 3 -G 1"
  logdir="/autofs/vast/lzgroup/Projects"
  endsig=""
elif [ "${site}" == "DZNE" ]; then
  callsig="nohup docker run --gpus ${gpu} -v /groups/ag-reuter/projects/master-theses/henschell:/henschell \
                          -v /home/henschell:/home/henschell --rm --name henschell_${logfile}_GPU${gpu##*=} \
                          -e \"PYTHONPATH=/henschell\" \
                           --user 4323:1275 --shm-size 8G super_res_seg_torchio:henschell_bash"
  endsig="> /groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/logs/${logfile} &"
else
  callsig="echo"
  endsig=""
fi

hdf5_train="${logdir}/FastInfantSurfer/hdf5_sets/validation_${sub}_${plane}.hdf5"
hdf5_val="${logdir}/FastInfantSurfer/hdf5_sets/validation_${sub}_${plane}.hdf5"

cmd="${callsig} python3 $base/SuperResSurfer/run_model.py \
  --cfg ${cfg} ${ext_aug} --opt aseg --opt children \
  MODEL.NUM_FILTERS 64 MODEL.KERNEL_H 3 MODEL.KERNEL_W 3 MODEL.MODEL_NAME ${net} MODEL.NUM_CLASSES ${num_classes} \
  LOG_DIR ${logdir}/FastInfantSurfer/experiments \
  DATA.LUT ${base}/FastInfantSurfer/configs/FastInfantSurfer_dHCP_LUT.tsv \
  DATA.PLANE $plane ${lat_aug} ${efa_aug} DATA.IMG_TYPE ${img_dtype} DATA.DIMENSION ${data_dim} \
  $opt_lr \
  TRAIN.NUM_EPOCHS ${epochs} TRAIN.BATCH_SIZE ${batchsize} TRAIN.CHECKPOINT_PERIOD ${ckptper} \
  MODEL.BASE_RES ${baseres} MODEL.HEIGHT ${mh} MODEL.WIDTH ${mw} \
  MODEL.OUT_TENSOR_WIDTH ${OTW} MODEL.OUT_TENSOR_HEIGHT ${OTH} DATA.PADDED_SIZE ${PDS} \
  DATA.PATH_HDF5_TRAIN ${hdf5_train} \
  DATA.PATH_HDF5_VAL ${hdf5_val} \
  EXPR_NUM ${ckpt} ${resume} ${endsig}"

echo $cmd
