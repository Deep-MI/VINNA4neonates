# IMPORTS
import numpy as np
import torch
import os
import logging
import nibabel as nib
import pandas as pd
import re
import argparse
import FastSurferCNN.data_loader.conform as conf
from FastSurferCNN.utils.arg_types import vox_size

from VINNA.eval import Inference
import VINNA.utils.metrics as metrics
from VINNA.utils.load_config import load_config
import VINNA.data_processing.utils.data_utils as du
from VINNA.data_processing.mapping import hcp_mapping_script as hms


##
# Global Variables
##
LOGGER = logging.getLogger("eval")
LOGGER.setLevel(logging.DEBUG)
#LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))

RES_LUT = {0.5: ["_min.nii.gz", "05"], 0.8: ["_08_dhcp.nii.gz", "08"], 1.0: [".nii.gz", "10"]}


##
# Processing
##
def set_up_cfgs(cfg, dict_args):
    cfg = load_config(cfg)
    cfg.OUT_LOG_DIR = dict_args["sd"] if dict_args["sd"] is not None else cfg.LOG_DIR
    cfg.OUT_LOG_NAME = dict_args["model_name"]
    cfg.TEST.BATCH_SIZE = dict_args["batch_size"]
    cfg.DATA.LUT = dict_args["lut"]

    if dict_args["outdims"] != 0:
        cfg.DATA.PADDED_SIZE = dict_args["outdims"]
    out_dims = round(dict_args["base_dim"] * dict_args["scale_output"]) if dict_args["scale_output"] != 1.0 and not dict_args["membership_interpol"] else cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_WIDTH = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    cfg.MODEL.OUT_TENSOR_HEIGHT = out_dims if out_dims > cfg.DATA.PADDED_SIZE else cfg.DATA.PADDED_SIZE
    return cfg


##
# Input array preparation
##
class RunMetricsData:

    def __init__(self, dict_args):
        """
        Init of run metrics. Reads in arguments from dictionary. Arguments used in setup:
          orig_name str: original string name
          sd str/None: output directory, set to LOG_DIR in cfg, if not defined
          model_name str: Name of model for storing purposes
          batch_size int: Batch size for inference
          fs bool: FreeSurfer segmentation is processed
          load_pred_from_disk bool: prediction already exists and should be loaded from disk

          # Model changes for inference
          scale_only bool: Only intensity scale input image, do not conform to LIA
          vox_size float/"min": voxel size to conform to
          outdims int: dimension of the output of the network (if set to 0, PADDED_SIZE from cfg is used)

          # If SuperRes, the image can be rescaled to a different size (FlexSurfer, Membership interpolation)
          scale_output float: rescaling factor for output in FlexSurfer or upsampling of softmax predictions. Default: 1.0
          membership_interpol bool: run upsampling of softmax predictions
          base_dim int: base dimension of the network (normally read from cfg, only needs to be changed, when
                        scale_output is defined)


          # Class options
          lut str: look-up table with classes to use
          combine bool: Combine certain classes to get estimates on e.g. complete cortex, WM, subcorticals

          # CFG options
          cfg_cor str: Config file for coronal model
          ckpt_cor str: Checkpoint path to coronal model
          cfg_sag str: Config file for sagittal model
          ckpt_sag str: Checkpoint path to sagittal model
          cfg_ax str: Config file for axial model
          ckpt_ax str: Checkpoint path to axial model

        :param dict_args:
        """
        self.subject_name = ""
        self.gt, self.gt_data, self.orig, self.orig_data = "", "", "", ""
        self.model_name, self.orig_filename = "", ""
        self.orig_name = dict_args["orig_name"]
        self.s = ""
        self.existing_sbjs = pd.Series(dtype=str)
        self.fs = dict_args["fs"]
        fs_labels = ["Left-Hippocampus", "Right-Hippocampus",  "Left-Amygdala", "Right-Amygdala",
                     "Left-Cerebral-Cortex", "Right-Cerebral-Cortex",
                     "Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter",
                     "Left-Cerebellum", "Right-Cerebellum"]
        fs_subcort = np.asarray([True, True, True, True, False, False, False, False, True, True])
        self.load_pred = dict_args["load_pred_from_disk"]

        #self.mem_interpol = args.membership_interpol
        self.scale_only = dict_args["scale_only"]
        self.vox_size = dict_args.get("vox_size", "min")

        # If SuperRes, the image can be rescaled to a different size (FlexSurfer, Membership interpolation)
        self.out_scale_factor = dict_args["scale_output"]
        self.membership_interpol = dict_args["membership_interpol"]

        self.lut = du.read_classes_from_lut(dict_args["lut"])
        self.labellist = set(self.lut["ID"].to_list())
        self.num_classes = len(self.lut["LabelName"])
        self.class_names = self.lut["LabelName"][1:] if not dict_args["fs"] else fs_labels
        exclude = ["Right-Cerebral-Cortex", "Left-Cerebral-Cortex",
                   "Right-Cerebral-White-Matter", "Left-Cerebral-White-Matter",
                   #"Cerebral-White-Matter", "Cerebral-Cortex"
                  ]
        self.labels = self.lut["ID"][1:]
        self.torch_labels = torch.from_numpy(self.lut["ID"].values)
        if dict_args["combine"]:
            self.combined_labels = {89: [6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],  # Right-GM
                                    90: [5, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],  # Left-GM
                                    91: [51, 53, 55, 57, 59, 61, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82],  # Right-WM,
                                    92: [52, 54, 56, 58, 60, 62, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81],  # Left-WM
                                    93: [42, 86], 94: [43, 87]}  # Right- and Left-Thalamus
            combined_names = exclude + ["Right-Thalamus", "Left-Thalamus"]
            self.class_names = np.hstack((self.class_names, combined_names))
            exclude.extend(['Right-Thalamus_low_intensity_part_in_T2', 'Left-Thalamus_low_intensity_part_in_T2',
                            'Right-Thalamus_high_intensity_part_in_T2', 'Left-Thalamus_high_intensity_part_in_T2'
                            ])

        else:
            self.combined_labels = None
        sub_list = [el for el in self.class_names if el not in exclude and el[-3:] != "_GM" and el[-3:] != "_WM"]
        self.range_sub = np.where(np.isin(self.class_names, sub_list), True, False) if not dict_args["fs"] else fs_subcort
        #print(f"Subcort list: {sub_list}, range_sub: {self.range_sub}")
        self.names = ["SubjectName", "Average", "Subcortical"]

        self.measure = ["dice_", "surfaceAverageHausdorff_"]
        self.save_list = []

        if not dict_args["load_pred_from_disk"]:
            cfg_cor = set_up_cfgs(dict_args["cfg_cor"], dict_args) if dict_args["cfg_cor"] is not None else None
            cfg_sag = set_up_cfgs(dict_args["cfg_sag"], dict_args) if dict_args["cfg_sag"] is not None else None
            cfg_ax = set_up_cfgs(dict_args["cfg_ax"], dict_args) if dict_args["cfg_ax"] is not None else None

            self.view_ops = {"coronal": {"cfg": cfg_cor,
                                         "ckpt": dict_args["ckpt_cor"]},
                         "sagittal": {"cfg": cfg_sag,
                                      "ckpt": dict_args["ckpt_sag"]},
                         "axial": {"cfg": cfg_ax,
                                   "ckpt": dict_args["ckpt_ax"]}}
            self.cfg_fin = cfg_cor if cfg_cor is not None else cfg_ax if cfg_ax is not None else cfg_sag
            self.ckpt_fin = dict_args["ckpt_cor"] if dict_args["ckpt_cor"] is not None \
                else dict_args["ckpt_ax"] if dict_args["ckpt_ax"] is not None \
                else dict_args["ckpt_sag"]
            self.model = Inference(self.cfg_fin, self.ckpt_fin)
            self.device = self.model.get_device()
            self.dim = self.model.get_max_size()
            self.views_to_use = [key for key in self.view_ops.keys() if self.view_ops[key]["cfg"] is not None]
        else:
            cfg = dict_args["cfg_cor"] if dict_args["cfg_cor"] is not None \
                else dict_args["cfg_ax"] if dict_args["cfg_ax"] is not None \
                else dict_args["cfg_sag"]
            self.cfg_fin = set_up_cfgs(cfg, dict_args)

        self.out_dir = os.path.join(self.cfg_fin.OUT_LOG_DIR, "eval_metrics")

    def get_torch_labels(self):
        return self.torch_labels

    def set_orig(self, orig_str):
        self.orig, self.orig_data = self.get_img(orig_str)
        # check and conform image
        print(f" Pre-Input image - VS: {self.orig.header.get_zooms()}, Dim: {self.orig_data.shape}")
        if not conf.is_conform(self.orig, conform_vox_size="min", check_dtype=True):
            if self.scale_only:
                if self.vox_size == "min":
                    print('Re-scaling image. No orientation change')
                    src_min, scale = conf.getscale(self.orig_data, 0, 255)
                    img = conf.scalecrop(self.orig_data, 0, 255, src_min, scale)
                    self.orig_data = np.uint8(np.rint(img))
                else:
                    print(f'Re-scaling image, scaling to {self.vox_size}. No orientation change')
                    from nibabel.freesurfer.mghformat import MGHHeader

                    conformed_vox_size, conformed_img_size = conf.get_conformed_vox_img_size(
                        self.orig, self.vox_size,
                    )
                    # limit image size to 320
                    if conformed_img_size > 256:
                        conformed_img_size = 256
                    # may copy some parameters if input was MGH format
                    h1 = MGHHeader.from_header(self.orig.header)

                    h1.set_data_shape(
                        [conformed_img_size, conformed_img_size, conformed_img_size, 1])
                    h1.set_zooms(
                        [conformed_vox_size, conformed_vox_size, conformed_vox_size]
                    )  # --> h1['delta']
                    h1["Mdc"] = [[-1, 0, 0], [0, 0, -1], [0, 1, 0]]
                    h1["fov"] = conformed_img_size * conformed_vox_size
                    h1["Pxyz_c"] = self.orig.affine.dot(
                        np.hstack((np.array(self.orig.shape[:3]) / 2.0, [1])))[:3]

                    # Here, we are explicitly using MGHHeader.get_affine() to construct the affine as
                    # MdcD = np.asarray(h1['Mdc']).T * h1['delta']
                    # vol_center = MdcD.dot(hdr['dims'][:3]) / 2
                    # affine = from_matvec(MdcD, h1['Pxyz_c'] - vol_center)
                    affine = h1.get_affine()

                    # from_header does not compute Pxyz_c (and probably others) when importing from nii
                    # Pxyz is the center of the image in world coords

                    src_min, scale = 0, 1.0
                    # get scale for conversion on original input before mapping to be more similar to
                    # mri_convert
                    if self.orig.get_data_dtype() != np.dtype(np.uint8):
                        src_min, scale = conf.getscale(np.asanyarray(self.orig.dataobj), 0, 255)

                    mapped_data = conf.map_image(self.orig, affine, h1.get_data_shape())

                    if self.orig.get_data_dtype() != np.dtype(np.uint8):
                        scaled_data = conf.scalecrop(mapped_data, 0, 255, src_min, scale)
                        # map zero in input to zero in output (usually background)
                        scaled_data[mapped_data == 0] = 0
                        mapped_data = scaled_data
                    self.orig_data = np.uint8(np.clip(np.rint(mapped_data), 0, 255))
                    self.orig = nib.MGHImage(self.orig_data, affine, h1)
                    self.orig.set_data_dtype(np.uint8)
            else:
                print(f'Re-scaling and Conforming image to {self.vox_size} size')
                self.orig = conf.conform(self.orig, conform_vox_size=self.vox_size)
                self.orig_data = np.asarray(self.orig.get_fdata(), dtype=np.uint8)

        # Calculate output dimensions for the image output (voxelsize, dimensions)
        self.output_shape = tuple([int(x * self.out_scale_factor) for x in self.orig_data.shape])
        self.output_zoom = tuple(np.array(self.orig.header.get_zooms()) / self.out_scale_factor)
        self.pred_affine = self.orig.affine

        print(f" Input image - VS: {self.orig.header.get_zooms()}, Dim: {self.orig_data.shape}")
        print(f"Output Image - VS: {self.output_zoom}, Dim: {self.output_shape}")

    def set_gt(self, gt_str, postprocess=False):
        self.gt, self.gt_data = self.get_img(gt_str)
        if postprocess:
            self.reorient_and_split_dhcp_gt()

    def reorient_and_split_dhcp_gt(self):
        # Split label 87 into two parts (intracranial GM split per hemi)
        split = hms.split_dhcp(self.gt_data, self.labellist)

        # Reorient from dhcp LAS output orientation to FreeSurfer conform LIA format
        las = {"swaps": [1, 2], "flips": (1)}
        reorient = np.swapaxes(split, las["swaps"][0], las["swaps"][1])
        self.gt_data = np.flip(reorient, axis=las["flips"])

    def set_subject(self, subject):
        self.subject_name = subject

    def set_model(self, plane):
        self.model.set_model(self.view_ops[plane]["cfg"])
        self.model.load_checkpoint(self.view_ops[plane]["ckpt"])
        self.device = self.model.get_device()
        self.dim = self.model.get_max_size()

    def set_filename(self, dirname):
        self.subject_name = os.path.basename(dirname)
        self.orig_filename = os.path.join(dirname, self.orig_name)

    def set_fname_sbj(self, sid, t1):
        self.subject_name = sid
        self.orig_filename = t1

    def set_modelname(self, modelname):
        self.model_name = modelname

    def get_gt(self):
        return self.gt, self.gt_data

    def get_orig(self):
        return self.orig, self.orig_data

    def check_if_sbj_needs_processing(self):
        return False if self.subject_name in self.existing_sbjs.values else True

    def get_prediction(self):
        # define the final prediction shape, this is identical to the orig shape, unless scale_out is defined
        # As this is currently not implemented, prediction probability can be the same for all
        if self.out_scale_factor == 1.0:
            out = torch.zeros(self.output_shape + (self.num_classes, ), dtype=torch.float).to(self.device)
        else:
            out = None

        for view in self.views_to_use:
            # set model
            LOGGER.info(f"Run {view} view agg")
            self.set_model(view)
            # Prediction gets overwritten in model.run call (views are added on top of each other); for super res
            # this does not work, so we add the prediction to out here.
            if view == self.views_to_use[0] and self.out_scale_factor != 1.0:
                out = self.single_view_inference(view)
            else:
                out += self.single_view_inference(view, out=out)
            #out = self.model.run(self.orig_filename, self.orig_data, self.orig.header.get_zooms(), out) #, intensity

        """
        To Do: How to correctly to inference with scaling and orig > scaled version 
        # Do membership interpolation, if scale_out is defined, but not using FlexSurfer
        """
        # Get hard predictions and map to freesurfer label space
        _, pred_prob = torch.max(out, 3)
        pred_prob = du.map_label2aparc_aseg(pred_prob.cpu(), self.torch_labels)

        # return numpy array
        return pred_prob #, intensity

    def crop_image_to_minsize(self, img):
        # Calculate correct min affine from orig affine, shape, output zoom and data shape (in case
        # ground truth data shape is different from inferred zoom shape)
        min_affine = nib.affines.rescale_affine(self.orig.affine, self.orig.shape, self.output_zoom,
                                                new_shape=self.gt_data.shape)
        ras_zero = np.asarray([0, 0, 0, 1])
        ras_zero_vox_min = np.linalg.inv(min_affine) @ ras_zero

        # Calculate affine the image has after rescaling according to the zoom-factor alone
        rescaled_affine = nib.affines.rescale_affine(self.orig.affine, self.orig.shape, self.output_zoom,
                                                     new_shape=self.output_shape)
        ras_zero_vox_rescale = np.linalg.inv(rescaled_affine) @ ras_zero
        start = (ras_zero_vox_rescale[:-1] - ras_zero_vox_min[:-1]).astype(int)
        stop = start + self.gt_data.shape
        pred_cut = img[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]

        # reset header and affine for saving the image in the correct format
        self.pred_affine = min_affine

        return pred_cut

    def get_shape_for_view(self, view):
        out_shape = list(self.output_shape + (self.num_classes, ))
        idx = {"sagittal": 0, "axial": 1, "coronal": 2}
        out_shape[idx[view]] = self.orig_data.shape[idx[view]]
        return tuple(out_shape)

    def single_view_inference(self, view, out=None):

        if self.out_scale_factor != 1.0 and not self.membership_interpol:
            # set up the prediction --> final prediction shape except for view dimension
            finshape = self.get_shape_for_view(view)
            out_full = torch.zeros(finshape, dtype=torch.float).to(self.device)
            out_res = self.cfg_fin.MODEL.BASE_RES/self.output_zoom[0]

        elif self.membership_interpol:
            # set up the prediction --> orig data shape for all dimensions
            out_full = torch.zeros(self.orig_data.shape + (self.num_classes,), dtype=torch.float).to(self.device)
            out_res = None
        else:
            out_full = out
            out_res = None

        # run the model; internally, predictions are cropped to orig shape (padded size -> out shape)
        out_full += self.model.run(self.orig_filename, self.orig_data, self.orig.header.get_zooms(), out_full,
                                  out_res=out_res)

        # rescale along view dimension (FlexSurfer) or along all dimensions (membership interpol), has to be done here,
        # because we do inference in 2D only (need to interpolate third dimension)
        # permute to channel first, unsqueeze to get four dimensional shape (Batch size, Channel, H, W, D)
        if self.out_scale_factor != 1.0 or self.membership_interpol:
            out_full = out_full.permute(3, 0, 1, 2).unsqueeze(0)
            out_full = torch.nn.functional.interpolate(out_full.cpu(), size=self.output_shape, mode="trilinear",
                                                       align_corners=False)
            # permute to channel last and squeeze batch dimension
            out_full = torch.squeeze(out_full, 0).permute(1, 2, 3, 0)
        return out_full

    @staticmethod
    def get_img(filename):
        img = nib.load(filename)
        data = np.asanyarray(img.dataobj)

        return img, data

    def save_img(self, save_as, data, data_header, dtype_img=np.int16):
        # Create output directory if it does not already exist.
        if not os.path.exists("/".join(save_as.split("/")[0:-1])):
            LOGGER.info("Output image directory does not exist. Creating it now...")
            os.makedirs("/".join(save_as.split("/")[0:-1]))
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()

        # correct header info (assert correct shape and zoom for super res/membership interpol)
        data_header.header.set_data_shape(self.output_shape)
        data_header.header.set_zooms(zooms=self.output_zoom)
        du.save_image_as_nifti(data_header.header, self.pred_affine, data, save_as, dtype_img)
        LOGGER.info("Successfully saved image as {}".format(save_as))

    def set_up_model_params(self, plane, cfg, ckpt):
        self.view_ops[plane]["cfg"] = cfg
        self.view_ops[plane]["ckpt"] = ckpt

    def set_load_pred(self, value):
        self.load_pred = value

    def instantiate_csv(self):
        # Instantiate csv-file
        meas_sub = "{}\t" * len(self.class_names)  # Placeholder for classes (as calculated in evaluate_metrics)
        title = "{}\t" * len(self.names)  # SubjectName, Average, Subcortical
        self.s = title + meas_sub + "{}\t{}\n"  # Net_type, SF at end
        val_header = title.format(*self.names) + meas_sub.format(*self.class_names) + "Net_type\tSF\n"

        for save_as in self.save_list:
            if not os.path.exists(save_as):
                with open(save_as, "w") as val_log:
                    val_log.write(val_header)
                LOGGER.info("Successfully created file {}".format(save_as))

    def get_existing_sbjs(self):
        dsc_filename = f"{self.out_dir}/dice_{self.model_name}.tsv"
        self.existing_sbjs = pd.read_csv(dsc_filename, sep="\t", names=["SubjectName"])

    def add_entry(self, measure, sf):
        assert self.subject_name != "" and self.model_name != "", f"Error: Subject and Modelname are not set!"
        for csv_file in self.save_list:
            meas = measure[csv_file.split("/")[-1].split("_")[0]]
            with open(csv_file, "a") as val_log:
                val_log.write(self.s.format(self.subject_name, np.nanmean(meas),
                                            np.nanmean(meas[self.range_sub]),
                                            *meas, self.model_name, sf))

    def add_entries(self, results):
        for result in results:
            mn = result["Net_type"]
            dsc, shd = f"{self.out_dir}/dice_{mn}.tsv", f"{self.out_dir}/surfaceAverageHausdorff_{mn}.tsv"
            dsca, dscs = result["dice"], result["surfaceAverageHausdorff"]
            with open(dsc, "a") as val_log_dsc:
                val_log_dsc.write(self.s.format(result["SubjectNames"], np.nanmean(dsca),
                                                np.nanmean(dsca[self.range_sub]), *dsca,
                                                result["Net_type"], result["SF"]))
            with open(shd, "a") as val_log_dsc:
                val_log_dsc.write(self.s.format(result["SubjectNames"], np.nanmean(dscs),
                                                np.nanmean(dscs[self.range_sub]), *dscs,
                                                result["Net_type"], result["SF"]))

    def set_names(self, new_names):
        self.names = new_names

    def set_range(self, range_total, range_sel):
        self.range_total = range_total
        self.range_sub = range_sel

    def set_save_list(self):
        self.save_list = [f"{self.out_dir}/{m}{self.model_name}.tsv" for m in self.measure]

    def get_save_list(self):
        return self.save_list

    def calculate_dsc(self, prediction):
        return tmf.dice_score(prediction, self.gt_data, reduction=None)

    def calculate_vs(self, prediction):
        return metrics.vs(prediction, self.gt_data, reduction=None)

    def calculate_hausdorff(self, prediction):
        return metrics.hd(prediction, self.gt_data, reduction=None)

    def evaluate(self, pred_data):
        return metrics.evaluate_metrics(self.gt_data, pred_data, self.labels, self.combined_labels, self.fs)
        #{"dice_": self.calculate_dsc(pred_data), "vs_": self.calculate_dsc(pred_data),
                #"surfaceAverageHausdorff_": self.calculate_dsc(pred_data)}


def determine_fnames(args, curr_idx, meta, col="Resolution", res_lut=RES_LUT):
    if meta is not None:
        res_info = res_lut[meta[col][curr_idx]]
        to_add = res_info[0]
        to_strip = -len(args.orig_name[len(args.orig_name.split("_")[0]):])
        pattern = re.compile("[0-9][0-9]")
        # Replace old resolution str (two numbers - 05, 08 or 10) with correct string based on res we did inference on
        # (stored in res_info[1])
        mn = pattern.sub(res_info[1], args.model_name) if meta is not None else args.model_name
        return args.gt_name[:to_strip] + to_add.replace("_dhcp", ""), args.orig_name[
                                                                      :to_strip] + to_add, f"aseg.{mn}.mgz", mn
    else:
        gtn = args.gt_name #.replace("_label", "-label")
        on = args.orig_name #.replace("_T", "-T")
        return gtn, on, f"aseg.{args.model_name}.mgz", args.model_name


def arg_setup():
    parser = argparse.ArgumentParser(description='Evaluation metrics')

    # 1. Options for input directories and filenames
    parser.add_argument('--gt_name', type=str, default="mri/aseg.mgz",
                        help="Default name for ground truth segmentations. Default: mri/aseg.filled.mgz")
    parser.add_argument('--orig_name', type=str, dest="orig_name", default='mri/orig.mgz', help="Name of orig input")
    parser.add_argument('--gt_dir', type=str, default=None,
                        help="Directory of ground truth (if different from orig input).")
    parser.add_argument('--csv_file', type=str, help="Csv-file with subjects to analyze (alternative to --pattern)",
                        default=None)
    parser.add_argument("--inter_rater_dir", type=str, default=None,
                        help="Directory with inter_rater results should be written.")
    parser.add_argument("--lut", type=str, default="/fastsurfervinn/configs/FastInfantSurfer_dHCP_full.tsv",
                        help="Path and name of LUT to use. Default expects it to be in /fastsurfer/vinn/configs")
    parser.add_argument("--substruct", action="append",
                        default=[1, 16],
                        help="Substructures to calculate DSC over.")

    # 2. Option for output directory inter_rater_dir
    parser.add_argument("--sd", type=str, default=None,
                        help="Directory in which evaluation results should be written. "
                             "Will be created if it does not exist")
    parser.add_argument("--sid", type=str, default=None,
                        help="Subject ID to run inference on")
    parser.add_argument("--single_img", default=False, action="store_true",
                        help="Run single image for testing purposes instead of entire csv-file")
    parser.add_argument("--tw", default=None, type=str,
                        help="T1 or T2 image to run inference on")

    # 3. Scale factor options or membership interpolation to ground truth seg shape
    parser.add_argument("--scale_output", type=float, default=1.0,
                        help="What resolution change should output have (e.g. from 1mm to 0.8 --> 0.8, "
                             "from 0.8 to 1mm --> 1.25). Default: 1.0 = no rescaling")
    parser.add_argument("--membership_interpol", action="store_true", default=False,
                        help="Run membership interpolation to scale_output on softmax predictions. "
                             "Assure to also change the --scale_output flag to the correct value "
                             "(what resolution change you want)")
    parser.add_argument("--base_res", type=float, default=1.0,
                        help="Base resolution (what resolution does input image have). Default: 1.0 (for 1mm)")
    parser.add_argument("--base_dim", type=int, default=256,
                        help="Base dimension (what dimension does input image have). Defaul: 256 (for 1mm)")
    parser.add_argument("--outdims", type=int, default=0,
                        help="Output dimension (what dimension should output image have). Default: 0 (for 1mm)")

    # 3. Checkpoint to load
    parser.add_argument('--ckpt_cor', type=str, help="coronal checkpoint to load")
    parser.add_argument('--ckpt_ax', type=str, default=None, help="axial checkpoint to load")
    parser.add_argument('--ckpt_sag', type=str, default=None, help="sagittal checkpoint to load")
    parser.add_argument('--model_name', type=str, default="FastSurferVINN_LatentAug",
                        help="Name with which prediction and csv-file will be stored")
    parser.add_argument('--crop', action="store_true", default=False,
                        help="Crop images to 256x256x256 prior to inference")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default=8")

    # 3. Options for evaluation measures
    parser.add_argument('--load_pred_from_disk', action='store_true', default=False,
                        help="Load prediction from disk (must already exist in this case)")
    parser.add_argument('--save_img', action='store_true', default=False, help="Save prediction as mgz on disk.")
    parser.add_argument('--metrics', action='store_true', default=False, help="Calculate metrics based on image.")
    parser.add_argument('--no_vol_measure', action='store_true', default=False,
                        help='disables calculation of volumetric overlap measures (DSC, JC, VS)')
    parser.add_argument('--no_hd_vol', action='store_true', default=False,
                        help='disables calculation of volumetric hausdorff distance (HD, AVG HD)')
    parser.add_argument('--no_hd_surf', action='store_true', default=False,
                        help='disables calculation of surface hausdorff distance (HD, AVG HD)')
    parser.add_argument('--symm_surf', action='store_true', default=False,
                        help="activates symmetric mesh-basd surface calculations (other measures are not calculated)")
    parser.add_argument('--rs_manual', action='store_true', default=False,
                        help="Calculate metrics only for Hippocampus, White Matter and Gray Matter")
    parser.add_argument('--split', action='store_true', default=False,
                        help="Split cortex classes befor saving prediction")
    parser.add_argument('--add_subject', action='store_true', default=False,
                        help="Add subject name as prefix to ground truth (for orig res inference)")
    parser.add_argument('--scale_only', action='store_true', default=False,
                        help="Only re-scale the input intensity image, do not change orientation (no conforming)")
    parser.add_argument('--vox_size', default="min", type=vox_size,
                        help="voxel size to conform to (default: min)")
    parser.add_argument('--fs', action='store_true', default=False,
                        help="Use aseg from freesurfer as <prediction>")
    parser.add_argument('--combine', action='store_true', default=False,
                        help="Combine labels in DSC calculations")

    # 4. CFG-file with default options for network
    parser.add_argument("--cfg_cor", dest="cfg_cor", help="Path to the config file",
                        default=None, type=str)
    parser.add_argument("--cfg_ax", dest="cfg_ax", help="Path to the axial config file",
                        default=None, type=str)
    parser.add_argument("--cfg_sag", dest="cfg_sag", help="Path to the sagittal config file",
                        default=None, type=str)

    # 5. Validation options with meta file
    parser.add_argument("--orig_dir", help="Base directory for orig input files", type=str,
                        default="/groups/ag-reuter/projects/datasets/dHCP/Data/")
    parser.add_argument("--val_run", action="store_true", default=False,
                        help="Run validation set (read meta info from csv-file)")
    parser.add_argument("--force_overwrite", action="store_true", default=False,
                        help="Run metrics on all subjects again, even if they were already exist in the csv-file")
    parser.add_argument("--num_processes", type=int, default=8, help="Number of processed for multi-processing DSC "
                                                                     "calculation")
    parser.add_argument("--multip", action="store_true", default=False,
                        help="Run multiprocessing for DSC processing (might speed things up)")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_setup()
    print("Starting run")
    # Get all subjects of interest
    if args.val_run:
        print("Loading csv-file")
        df = pd.read_csv(args.csv_file, sep="\t")
        df["CorrPath"] = args.orig_dir + df["SubjectFix"]
        s_dirs = df["CorrPath"].values

    elif not args.single_img:
        with open(args.csv_file, "r") as s_dirs:
            s_dirs = [line.strip() for line in s_dirs.readlines()]
        df = None

    LOGGER.info("Output will be stored in: {}".format(args.sd))

    vol_meas = not args.no_vol_measure
    hd_vol = not args.no_hd_vol
    hd_surf = not args.no_hd_surf

    if args.symm_surf:
        vol_meas, hd_vol, hd_surf = False, False, False

    LOGGER.info("Do:\nDSC, JC, VS: {}\nHD, AVG HD in Volume: {}\n"
                "HD, AVG HD on surf: {}\nSurf_symm: {}\n".format(vol_meas, hd_vol, hd_surf, args.symm_surf))

    LOGGER.info("Ground truth: {}, Origs: {}".format(args.gt_name, args.orig_name))

    # Create output directory if it does not already exist.
    if args.sd is not None and not os.path.exists(args.sd):
        LOGGER.info("Output directory does not exist. Creating it now...")
        os.makedirs(args.sd)

    # create dict from args
    args_as_dict = vars(args)

    # Set Up Model
    eval = RunMetricsData(args_as_dict)

    if not args.single_img:
        for i in range(len(s_dirs)):
            subject = s_dirs[i]
            base_sbj = os.path.basename(subject)
            dir_sbj = os.path.dirname(subject) if not os.path.isdir(subject) else subject
            if base_sbj == "anat":
                base_sbj = os.path.basename(os.path.dirname(subject))
                dir_sbj = os.path.join(args.gt_dir, base_sbj)
            print(f"Running on subject {subject}, directory {dir_sbj}")
            # Set orig and gt for testing now
            gt_name, orig_name, segname, modelname = determine_fnames(args, curr_idx=i, meta=df)

            if args.add_subject:
                gts = os.path.join(dir_sbj, base_sbj + gt_name)
                #origs = os.path.join(dir_sbj, base_sbj + orig_name)
            else:
                gts = os.path.join(dir_sbj, gt_name)
                #origs = os.path.join(dir_sbj, orig_name)

            origs = os.path.join(dir_sbj, orig_name)
            postprocessing_dhcp = True if gt_name.split(".")[0].endswith("all_labels_08") else False

            try:
                eval.set_gt(gts, postprocessing_dhcp)
                eval.set_modelname(modelname)
                eval.set_orig(origs)
                eval.set_subject(base_sbj)
            except FileNotFoundError as e:
                print(f"FileNotFound: {e}")
                continue

            if args.sd is not None:
                if gt_name[:3] == "mri":
                    pred_name = os.path.join(args.sd, base_sbj, "mri", segname)
                else:
                    pred_name = os.path.join(args.sd, base_sbj, segname)
                if not os.path.exists(os.path.join(args.sd, base_sbj)):
                    os.makedirs(os.path.join(args.sd, base_sbj))
            else:
                pred_name = os.path.join(subject, "aseg.mgz") if args.fs else os.path.join(subject, "mri", segname)

            # Run model
            if not args.load_pred_from_disk:
                pred_data = eval.get_prediction()
            else:
                try:
                    pred_data = np.asanyarray(nib.load(pred_name).dataobj)
                except FileNotFoundError as e:
                    print(f"FileNotFound: {e}")
                    continue

            if args.save_img:
                pred_data = eval.crop_image_to_minsize(pred_data)
                orig, orig_data = eval.get_orig()
                eval.save_img(pred_name, pred_data, orig)

            if args.metrics:
                if eval.check_if_sbj_needs_processing() and not args.force_overwrite:
                    # Do not rerun processing if sbj already exists in list
                    eval.set_save_list()
                    eval.instantiate_csv()
                    result = eval.evaluate(pred_data)
                    orig, orig_data = eval.get_orig()
                    eval.add_entry(result, orig.header.get_zooms()[0])

    else:
        # Set orig and gt for testing now
        subject = os.path.join(args.sd, args.sid) # s_dirs[1]
        LOGGER.info(f"Run model on {subject}")

        eval.set_orig(args.tw)
        eval.set_subject(args.sid)
        eval.set_fname_sbj(args.sid, args.tw)
        eval.set_modelname(args.model_name)
        pred_name = os.path.join(subject, "mri", "aseg." + args.model_name + ".mgz")

        # Run model
        if not args.load_pred_from_disk:
            pred_data = eval.get_prediction()
        else:
            pred_data = np.asanyarray(nib.load(pred_name).dataobj)

        orig, orig_data = eval.get_orig()
        gt, gt_data = eval.get_gt()

        if args.save_img:
            if not os.path.exists(os.path.dirname(pred_name)):
                os.makedirs(os.path.dirname(pred_name))
            eval.save_img(pred_name, pred_data, orig)

        # calculate metrics and save to file
        if args.metrics:
            gts = os.path.join(subject, os.path.basename(subject) + args.gt_name) if args.add_subject else os.path.join(
                subject,
                args.gt_name)
            eval.set_gt(gts)
            eval.set_save_list()
            LOGGER.info(eval.get_save_list())
            eval.instantiate_csv()
            result = eval.evaluate(pred_data)
            LOGGER.info(result)
            eval.add_entry(result, orig.header.get_zooms()[0])
