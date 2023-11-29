# NeonateVINNA Docker Image Creation

Within this directory we currently provide a Dockerfiles for the segmentation with the NeonateVINNA.
Note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details).  

### Build GPU NeonateVINNA Image

In order to build your own Docker image for NeonateVINNA (on GPU) simply execute the following command after traversing into the *Docker* directory: 

```bash
cd ..
docker build --rm=true -t neonatevinna:gpu-beta -f ./Docker/Dockerfile .
```

For running the analysis, you can use the single_img_inference.sh script in the run_scripts directory (Example 1) or directly use the docker call (Example 2) for which you need to specify more options by yourself currently.

### Example 1: Single image inference the simple way
```bash
chmod +x ./NeonateVINNA/run_scripts/single_img_inference.sh
./NeonateVINNA/run_scripts/single_img_inference.sh --tw <input_image> --id <input_directory> --sd <output directory> --sid <subject id> --mode <input image modality> [OPTIONS]
```
##### NeonateVINNA Flags
Requires flags:
* The `--id` points to the data input directory where the image you want to analyse resides on your computer (e.g./home/my_mri_data)
* The `--tw` points to the MRI image to analyse (path starts from whatever you put as the input_directory with flag --id 
             Example: data in /home/my_mri_data/subject1/anat/raw/t2w_orig.nii.gz and --id set as /home/my_mri_data --> --tw subject1/anat/raw/t2w_orig.nii.gz
             Example: data in /home/my_mri_data/subject1/anat/raw/t2w_orig.nii.gz and --id set as /home/my_mri_data/subject1/anat/raw --> tw t2w_orig.nii.gz
* The `--sd` points to the output directory on your computer (where you want the results to go)
* The `--sid` is the subject ID name (output folder name, e.g.\ --sid subject1; a folder with the name will be created in the output directory you specified if it does not exist)

Optional flags:
* The `--mode <image modality>` defines the input image modality. Either T2 (default) or T1.
* The `--augS <augmentation>` defines the image augmentation to use. One of _RotTLScale, _LatentAug, _LatentAugPlus (default)
* The `--net <network for inference>` defines which network to use for inference: FastSurferDDB or FastSurferVINN (default)
* The `--add <additional arguments>` defines additional arguments to add to the call (i.e. --outdims 320 for 0.5 mm images)

The output of the analysis will be stored in <output directory>/<subject id>/mri/aseg.<model>.mgz.

### Example 2: Single image inference the direct way
```bash
# You need to specify all variables with a dollar sign by yourself!
docker run --gpus device=${gpu} --name NeonateVINNA  \
            --rm --user $(id -u):$(id -g) \
            -v $sd:$sd \
            -v $id:$id \
            --shm-size 8G neonatevinna:gpu-beta \
            nohup python3 /NeonateVINNA/VINNA/run_validation.py --tw ${id}/${orig} \
                                         --sd ${sd} --sid ${sid}\
                                         --model_name ${mn}  \
                                         --ckpt_cor /NeonateVINNA/experiments/checkpoints/${net}/${augS}/${cfg}_coronal/Best_training_state.pkl \
                                         --cfg_cor /NeonateVINNA/experiments/config/${net}/${augS}/${cfg}_coronal/config.yaml \
                                         --ckpt_ax /NeonateVINNA/experiments/checkpoints/${net}/${augS}/${cfg}_axial/Best_training_state.pkl \
                                         --cfg_ax /NeonateVINNA/experiments/config/${net}/${augS}/${cfg}_axial/config.yaml \
                                         --ckpt_sag /NeonateVINNA/experiments/checkpoints/${net}/${augS}/${cfg}_sagittal/Best_training_state.pkl \
                                         --cfg_sag /NeonateVINNA/experiments/config/${net}/${augS}/${cfg}_sagittal/config.yaml  \
                                         --combine --orig_name mri/orig.mgz \
                                         --lut /NeonateVINNA/experiments/LUTs/FastInfantSurfer_dHCP_full_LUT.tsv \
                                         --batch_size 1 --single_img --save_img
```

##### Docker Flags:
* `--gpus`: This flag is used to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus "device=0,1,3"`.
* `-v`: This commands mount your data, and output directory into the docker container. Inside the container these are visible under the name following the colon (in this case they are the same names you need to define (e.g. id=/home/my_mri_data, sd=/home/vinna_analysis)).
* `--rm`: The flag takes care of removing the container once the analysis finished. 
* `-d`: This is optional. You can add this flag to run in detached mode (no screen output and you return to shell)
* `--user $(id -u):$(id -g)`: This part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root which is strongly discouraged.

##### NeonateVINNA Flags:
* The `--tw` points to the MRI image to analyse (path starts from whatever you put as the input_directory with flag --id
* The `--sd` points to the output directory on your computer (where you want the results to go)
* The `--sid` is the subject ID name (output folder name, e.g.\ --sid subject1; a folder with the name will be created in the output directory you specified if it does not exist)
* The `--model_name` is the name of the model you run. The generated prediction will have this name as an suffix (aseg.$model_name.mgz)
* The `--ckpt_*` defines the path to the checkpoints of the pre-trained models. There are three of them, one for each anatomical plane (*=cor is coronal view, *=ax is the axial view, *=sag is the sagittal view))
* The `--cfg_*` defines the path to the config files of the models. There are three flags, one for each anatomical plane (akin to the checkpoints)

Note, that for the docker command the paths following `--tw`, `--sd`, the config files, and checkpoints are __inside__ the container, not global paths on your system, so they should point to the places where you mapped these paths above with the `-v` arguments. 
