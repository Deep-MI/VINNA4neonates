# IMPORTS
import nibabel as nib
import numpy as np
import argparse
import pandas as pd
import glob
from os.path import join as opj
from os.path import basename as opb
from scipy.ndimage import gaussian_filter

MAPPINGS = {6: [6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
            5: [5, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],
            8: [52, 54, 56, 58, 60, 62, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81],
            7: [51, 53, 55, 57, 59, 61, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82],
            9: [43, 87],
            10: [42, 86]
            }

HCP_TO_FS = {2: 22, 3: 15, 41: 25, 42: 18, 43: 154, 4: 151}


def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='dHCP reduction mapping')

    parser.add_argument("--indir", type=str,
                        help="Input directory with subject folders and files to map"
                        )
    parser.add_argument("--outdir", type=str,  default=None,
                        help="Output directory to save mappes images. Default = None (save in input dir)")

    parser.add_argument("--origdir", type=str,
                        help="DHCP-rel3 directory to load ribbon images from.")


    parser.add_argument("--in_label_name", type=str,  default="_desc-drawem87_dseg.nii.gz",
                        help="Suffix of label to read in and map (default = _desc-drawem87_dseg.nii.gz)")

    parser.add_argument("--hcp_name_in", type=str, default=None,
                        help="Suffix of hcp label to read in and use to mask freesurfer segmentation"
                             " (default = _mapped23_dseg.nii.gz)")

    parser.add_argument("--name_orig_in", type=str, default=None,
                        help="Suffix of hcp label to read in and use to mask freesurfer segmentation"
                             " (default = _desc-drawem87_dseg.nii.gz)")

    parser.add_argument("--out_label_name", type=str, default="_mapped23_dseg.nii.gz",
                        help="Suffix of mapped label to save img under (default = -mapped_23_dseg.nii.gz)")

    parser.add_argument("--csv_file", default=None, type=str,
                        help="Csv-file with subjects to analyse (only subject names)")

    parser.add_argument("--pattern", default="*", type=str,
                        help="Pattern to search for subjects in indir. Default = all subjects in directory")

    parser.add_argument("--freesurfer", action="store_true", default=False,
                        help="Map and combine dHCP and FreeSurfer labels")

    parser.add_argument("--fix", action="store_true", default=False,
                        help="Fix wrong assignment of label 85 to BG in dHCP 23 classes mapped segmentation")

    parser.add_argument("--split_dhcp", action="store_true", default=False,
                        help="Fix wrong assignment of label 85 to BG in dHCP 87 classes mapped segmentation")

    parser.add_argument("--label_list", type=str, default="/groups/ag-reuter/projects/master-theses/henschell/FastInfantSurfer/configs/FastInfantSurfer_dHCP_full_LUT.tsv",
                        help="Meta file with label information")

    return parser.parse_args()


def load_image(fstr):
    """
    Load image with nibabel. Return array data and img object
    :param fstr: MRI image path and name
    """
    img = nib.load(fstr)
    return img, np.asanyarray(img.dataobj, dtype=np.int16)


def save_image(image_to_save, affine, header, save_as, dtype_set=np.int16):
    """
        Function to save a given file as a nifti-image
        :param ndarray image_to_save: image with dimensions (height, width, depth) to be saved in nifti format
        :param ndarray affine: affine information
        :param ndarray header: header information
        :param str save_as: name and directory where the nifti should be stored
        .param dtype dtype_set: dtype under which image should be saved (int by default)
        :return: void
        """
    header.set_data_dtype(dtype_set)
    nifti_new = nib.nifti1.Nifti1Pair(image_to_save, affine, header)
    nifti_new.set_data_dtype(np.dtype(dtype_set))  # not uint8 if aparc!!! (only goes till 255)
    nib.nifti1.save(nifti_new, save_as)


def map_hcp(img, dhcp_labels, mapping=MAPPINGS):
    """
    Mapping 87 label HCP image to reduced form similar to freesurfer (24 classes, hemi split)
    :param np.ndarray img: image to map
    :return np.ndarray: mapped/reduced image
    """
    dims = img.shape
    if not set(np.unique(img)) == set(dhcp_labels):
        img = split_dhcp(img, dhcp_labels)
    assert img.shape == dims
    for new_lab, lab in mapping.items():
        mask = np.in1d(img, lab).reshape(img.shape)
        img[mask] = new_lab
    assert img.shape == dims
    return img


def split_dhcp(dhcp, dhcp_labels, mapping=MAPPINGS, intracranial_label=85):
    assert set(np.unique(dhcp)).issubset(dhcp_labels)
    label_list_left = mapping[7] + [47] + mapping[9]
    label_list_right = mapping[8] + [46] + mapping[10]
    white_left = np.in1d(dhcp, label_list_left).reshape(dhcp.shape)
    white_right = np.in1d(dhcp, label_list_right).reshape(dhcp.shape)

    # Mask consists of labels for Thalamus, WM and Lentiform Nucleus
    return split_label_based_on_white(dhcp, [intracranial_label], white_left, white_right, add=3)


def fix_map_hcp(orig, fs=None, label=85):
    """
    Function to fix wrong alignment of label 85 (gray matter) to BG.
    In addition, label is split into left-right hemi
    :param np.ndarray orig: original dHCP label with 87 classes
    :param np.ndarray/None fs: freesurfer recon-surf to assign images to left/right hemi
    :param int label: label to resplit (default=85 for intracranial background)
    :return:
    """
    img = map_hcp(orig)
    mask = (orig == label)
    if fs is None:
        # Mask consists of labels for Thalamus, WM and Lentiform Nucleus
        mask_left = (img == 9) | (img == 47) | (img == 7)
        mask_right = (img == 10) | (img == 46) | (img == 8)
        img = split_label_based_on_white(img, [label], white_idl=mask_left, white_idr=mask_right, add=3)
    else:
        img[mask] = np.where(fs[mask] < 40, label-1, label)

    return img


def split_label_based_on_white(seg, label, left_side, right_side, add=-1):
    """
    Splot cortex labels to completely de-lateralize structures
    :param np.ndarray seg: anatomical segmentation to split
    :param list(int) label: labels to split. Defines RH label. LH label = RH - 1
    :param np.ndarray(bool) left_side: label mask for left hemisphere
    :param np.ndarray(bool): right_side label mask for right hemisphere
    :param int add: what to add to create left label from right (default = -1)
    :return np.ndarray: re-lateralized segmentation
    """

    # Get probability for left and right hemi by dilating WM labels
    aseg_lh = gaussian_filter(1000 * np.asarray(left_side, dtype=float), sigma=3)
    aseg_rh = gaussian_filter(1000 * np.asarray(right_side, dtype=float), sigma=3)

    lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                            axis=3)

    # Loop over classes and assign left or right label based on prob
    for prob_class_rh in label:
        prob_class_lh = prob_class_rh + add
        mask_lh = ((seg == prob_class_lh) | (seg == prob_class_rh)) & (lh_rh_split == 0)
        mask_rh = ((seg == prob_class_lh) | (seg == prob_class_rh)) & (lh_rh_split == 1)

        seg[mask_lh] = prob_class_lh
        seg[mask_rh] = prob_class_rh

    return seg


def fuse_aseg_hcp(aseg, hcp, mapping=HCP_TO_FS):
    """
    Fuse aseg and mapped hcp-image to one. Keep subcortical
    predictions from freesurfer and take cortex and WM labels from
    hcp (counteracts undersegmentation and skull strip errors).
    :param np.ndarray aseg: aseg segmentation (infant FreeSurfer)
    :param np.ndarray hcp: dhcp segmentation
    :param dict(int) mapping: mapping between freesurfer (key) and
    hcp labels (valules)
    :return:
    """
    # overwrite cortex and white-matter with WM
    for fs_lab, hcp_lab in mapping.items():
        aseg[hcp == hcp_lab] = fs_lab

    # Where aseg cortex/WM, but hcp CSF = Background
    ctx_wm_mask = np.in1d(aseg, [2, 41]).reshape(aseg.shape)
    hcp_csf_mask = np.in1d(hcp, [255]).reshape(hcp.shape)
    aseg[ctx_wm_mask & hcp_csf_mask] = 0
    return aseg


def process(subjects, args, label_ids=None):
    """
    Processing loop to map all images listed in subjects to 23 classes fs-like space.
    :param str indir: directory with subjects to analyze
    :param list(str) subjects: subjects to analyse
    :param str name_label_in: Name of label file
    :param str name_label_out: Name of mapped label file
    :param str hcp_name_in: Name of hcp label
    :param bool freesurfer: Indicate if mapping includes freesurfer labels or not
    :param str outdir: output directory in which new files will be stored. Default=None (save in input directoy)
    """
    for sbj in subjects:
        print(f"Running on subject {sbj} in {args.indir}")
        if not args.freesurfer:
            print("FreeSurfer processing")
            label, data = load_image(opj(args.indir, sbj, sbj + args.in_label_name))
            outname = opj(sbj, sbj + args.out_label_name)
            mapped_label = map_hcp(data, label_ids)
        else:
            outname = opj(sbj, args.out_label_name)
            #print("Aseg", data.shape, label.affine, data.dtype, "HCP", hcp_data.shape, hcp_label.affine, hcp_data.dtype)

            if args.fix:
                print("Fix processing")
                parts = sbj.split("_")
                if len(parts) < 2:
                    parts = sbj.split("ses-")
                    parts[1] = "ses-" + parts[1]

                #label, data = load_image(opj(args.origdir, parts[0], parts[1], "anat", sbj + args.in_label_name))
                outname = opj(sbj, sbj + args.out_label_name)
                label, data = load_image(opj(args.indir, sbj, parts[0] + "_" + parts[1] + args.name_orig_in))
                mapped_label = map_hcp(data, label_ids)
                assert data.shape == mapped_label.shape
                print(data.shape, mapped_label.shape)
            elif args.split_dhcp:
                outname = opj(sbj, sbj + args.out_label_name)
                label, data = load_image(opj(args.indir, sbj, sbj + args.name_orig_in))
                mapped_label = split_dhcp(data, label_ids)

            else:
                label, data = load_image(opj(args.indir, sbj, args.in_label_name))
                hcp_label, hcp_data = load_image(opj(args.indir, sbj, sbj + args.hcp_name_in))
                mapped_label = fuse_aseg_hcp(data, hcp_data, mapping=HCP_TO_FS)

        if args.outdir is None:
            save_as = opj(args.indir, outname)
        else:
            save_as = opj(args.outdir, outname)
        save_image(mapped_label, label.affine, label.header, save_as)


if __name__ == "__main__":
    args = setup_options()

    if args.csv_file is not None:
        with open(args.csv_file, "r") as f:
            subjects = [opb(sbj.strip()) for sbj in f.readlines()]
    else:
        subjects = [opb(sbj) for sbj in glob.glob(opj(args.indir, args.pattern))]
    if args.label_list is not None:
        labels = pd.read_csv(args.label_list, sep="\t")
        lab = set(labels["ID"].to_list())
    else:
        lab = None
    process(subjects, args, label_ids=lab)

