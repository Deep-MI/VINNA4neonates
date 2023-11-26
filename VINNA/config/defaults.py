from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# Name of model
_C.MODEL.MODEL_NAME = ""

# Number of classes to predict, including background
_C.MODEL.NUM_CLASSES = 34

# Loss function, combined = dice loss + cross entropy, combined2 = dice loss + boundary loss
_C.MODEL.LOSS_FUNC = "combined"

# Filter dimensions for DenseNet (all layers same)
_C.MODEL.NUM_FILTERS = 71

# Filter dimensions for Input Interpolation block (currently all the same)
_C.MODEL.NUM_FILTERS_INTERPOL = 32

# Type of additional operation on interpolation block output feature maps ("no", "simple", "matmul", "add")
_C.MODEL.INTERPOL_EXTRA = "no"

# Number of UNet layers in Basenetwork (including bottleneck layer!)
_C.MODEL.NUM_BLOCKS = 5

# Number of input channels (slice thickness)
_C.MODEL.NUM_CHANNELS = 7

# Number of branches for attention mechanism
_C.MODEL.NUM_BRANCHES = 5

# Height of convolution kernels
_C.MODEL.KERNEL_H = 3

# Width of convolution kernels
_C.MODEL.KERNEL_W = 3

# Depth of convolution kernels
_C.MODEL.KERNEL_D = 3

# size of Classifier kernel
_C.MODEL.KERNEL_C = 1

# Stride during convolution
_C.MODEL.STRIDE_CONV = 1

# Stride during pooling
_C.MODEL.STRIDE_POOL = 2

# Size of pooling filter
_C.MODEL.POOL = 2

# The height of segmentation model (after interpolation layer)
_C.MODEL.HEIGHT = 256

# The width of segmentation model
_C.MODEL.WIDTH = 256

# The base resolution of the segmentation model (after interpolation layer)
_C.MODEL.BASE_RES = 1.0

# Interpolation mode for up/downsampling in Flex networks
_C.MODEL.INTERPOLATION_MODE = "bilinear"

# Crop positions for up/downsampling in Flex networks
_C.MODEL.CROP_POSITION = "top_left"

# Out Tensor dimensions for interpolation layer
_C.MODEL.OUT_TENSOR_WIDTH = 320
_C.MODEL.OUT_TENSOR_HEIGHT = 320

# Flag, for smoothing testing (double number of feature maps before/after interpolation block)
_C.MODEL.SMOOTH = False

# Flag for elastic deformation
_C.MODEL.ELASTIC_DEFORMATION = False

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# input batch size for training
_C.TRAIN.BATCH_SIZE = 16

# how many batches to wait before logging training status
_C.TRAIN.LOG_INTERVAL = 50

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.RESUME = False

# The experiment number to resume from
_C.TRAIN.RESUME_EXPR_NUM = "Default"

# number of epochs to train
_C.TRAIN.NUM_EPOCHS = 30

# number of steps (iteration) which depends on dataset
_C.TRAIN.NUM_STEPS = 10

# To fine tune model or not
_C.TRAIN.FINE_TUNE = False

# checkpoint period
_C.TRAIN.CHECKPOINT_PERIOD = 2

# number of worker for dataloader
_C.TRAIN.NUM_WORKERS = 3

# Flag to disable or enable Early Stopping
_C.TRAIN.EARLY_STOPPING = True

# Mode for early stopping (min = stop when metric is no longer decreasing, max = stop when mwtric is no longer increasing)
# We use mIOU for tracking, so default mode should be max
_C.TRAIN.EARLY_STOPPING_MODE = "max"

# Patience = Number of epochs to wait before stopping
_C.TRAIN.EARLY_STOPPING_PATIENCE = 10

# Wait = NUmber of epochs before starting early stopping check
_C.TRAIN.EARLY_STOPPING_WAIT = 10

# Delta = change below which early stopping starts (previous - current < delta = stop)
_C.TRAIN.EARLY_STOPPING_DELTA = 0.00001

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# input batch size for testing
_C.TEST.BATCH_SIZE = 16

# ---------------------------------------------------------------------------- #
# Data options
# ---------------------------------------------------------------------------- #

_C.DATA = CN()

# path to training hdf5-dataset
_C.DATA.PATH_HDF5_TRAIN = ""

# path to validation hdf5-dataset
_C.DATA.PATH_HDF5_VAL = ""

# The plane to load ['axial', 'coronal', 'sagittal']
_C.DATA.PLANE = "coronal"

# The image type to load ['t2_image', 'image']; image = T1
_C.DATA.IMG_TYPE = "image"

# Which classes to use and meta info
_C.DATA.CLASS_OPTIONS = ["aseg", "aparc"]
_C.DATA.LUT = '/groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/configs/FastInfantSurfer_dHCP_LUT.tsv'
_C.DATA.META_INFO = '/groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/data/dataset_split_large_validation_t1t2_meta_slices.tsv'
_C.DATA.COMBINED_LABELS = True

# Available size for dataloader
# This for the multi-scale dataloader
_C.DATA.SIZES = [128, 183, 257, 256, 311, 320]

# the size that all inputs are padded to
_C.DATA.PADDED_SIZE = 320

# Augmentations
_C.DATA.AUG = ["Flip", "Elastic", "Scaling", "Rotation", "Translation", "RAnisotropy", "BiasField", "RGamma"]
_C.DATA.AUGNAME = "NoAug"
_C.DATA.SYNTHSEG_LUT = "/groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/configs/synthseg_infant_test.tsv"
# Trade off (1 - X = % SYNTHSEG), larger = less Synthseg, smaller = more --> 1 = no synthseg, 0 = only synthseg
_C.DATA.SYNTHSEG_TRADEOFF = 1.0

# Use latent space affines
_C.DATA.LATENT_AFFINE = False

# Use upright affine
_C.DATA.UPR_AFFINE = False

# Range of rotation angle (random rotations +/- in this range)
_C.DATA.Rot_ANGLE_MIN = 0
_C.DATA.Rot_ANGLE_MAX = 180

# Range of translation shift (random TL +/- in this range); pytorch = normed [-1, 1]
_C.DATA.TRANSLATION_MIN = 0.2  # for 256 image = 25.6
_C.DATA.TRANSLATION_MAX = 0.4  # for 256 image = 51,2

# Dimension to train (2D or 3D)
_C.DATA.DIMENSION = 2

# Elastic Deformation Parameters
_C.DATA.NUM_CTRL_POINTS = (7, 7)
_C.DATA.MAX_DISPLACEMENT = (8, 8)
_C.DATA.LOCKED_BORDERS = 2

# Mix Batches with T1 and T2 images (False by default, use only image type defined in DATA.IMG_TYPE)
_C.DATA.MIX_T1_T2 = False

# ---------------------------------------------------------------------------- #
# DataLoader options (common for test and train)
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()

# Number of data loader workers
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# EvalMetrics options
# ---------------------------------------------------------------------------- #
_C.EVALMETRICS = CN()
_C.EVALMETRICS.LUT = "/groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/configs/FastInfantSurfer_LUT.tsv"
_C.EVALMETRICS.SUBSTRUCT = [1, 20]

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()

# Base learning rate.
_C.OPTIMIZER.BASE_LR = 0.01

# Learning rate scheduler, step_lr, cosineWarmRestarts, reduceLROnPlateau
_C.OPTIMIZER.LR_SCHEDULER = "step_lr"

# Multiplicative factor of learning rate decay in step_lr
_C.OPTIMIZER.GAMMA = 0.3

# Period of learning rate decay in step_lr
_C.OPTIMIZER.STEP_SIZE = 5

# minimum learning in cosine lr policy and cosineWarmRestarts
_C.OPTIMIZER.ETA_MIN = 0.0001

# number of iterations for the first restart in cosineWarmRestarts
_C.OPTIMIZER.T_ZERO = 10

# A factor increases T_i after a restart in cosineWarmRestarts
_C.OPTIMIZER.T_MULT = 2

# factor by which learning rate will be reduce (new_lr = lr*factor, default=0.1)
_C.OPTIMIZER.FACTOR = 0.1

# number of epochs to wait before lowering lr (default=5)
_C.OPTIMIZER.PATIENCE = 5

# Threshold for measuring new optimum (default=1e-4)
_C.OPTIMIZER.THRESH = 0.0001

# Number of epochs to wait before resuming normal operation (default=0)
_C.OPTIMIZER.COOLDOWN=0

# min mode = lr will be reduced when monitored stops decreasing
# max mode = lr will be reduced when monitored quantity stops increasing
# Default: ‘min’.
_C.OPTIMIZER.MODE="min"

# Momentum
_C.OPTIMIZER.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIMIZER.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIMIZER.NESTEROV = True

# L2 regularization
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4

# Optimization method [sgd, adam, adamW]
_C.OPTIMIZER.OPTIMIZING_METHOD = "adamW"

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use
_C.NUM_GPUS = 1

# log directory for run
_C.LOG_DIR = "./experiments"

# experiment number
_C.EXPR_NUM = "Default"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Wandb set ups
_C.WANDB_PROJECT = "BabySurfer"

# Summary and config log paths
_C.SUMMARY_PATH = "./experiments/summary"
_C.CONFIG_LOG_PATH = "./experiments/config"

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()