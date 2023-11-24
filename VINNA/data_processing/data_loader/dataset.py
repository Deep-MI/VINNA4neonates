import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchio as tio
import time

import sys
sys.path.append("/groups/ag-reuter/projects")
sys.path.append("/groups/ag-reuter/projects/master-theses/henschell")
import NeonateVINNA.data_processing.utils.data_utils as du
import SuperResSurfer.SuperResSegm.utils.logging_utils as logging

logger = logging.get_logger(__name__)


# Operator to load imaged for inference
class MultiScaleOrigDataThickSlices(Dataset):
    """
    Class to load MRI-Image and process it to correct format for network inference
    """
    def __init__(self, img_filename, orig_data, orig_zoom, cfg, gn_noise=0, transforms=None):
        self.img_filename = img_filename
        self.plane = cfg.DATA.PLANE
        self.slice_thickness = cfg.MODEL.NUM_CHANNELS//2
        self.base_res = cfg.MODEL.BASE_RES
        self.gn_noise = gn_noise

        if self.plane == "sagittal":
            orig_data = du.transform_sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Sagittal with input voxelsize {}".format(self.zoom))

        elif self.plane == "axial":
            orig_data = du.transform_axial(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Axial with input voxelsize {}".format(self.zoom))

        else:
            self.zoom = orig_zoom[:2]
            logger.info("Loading Coronal with input voxelsize {}".format(self.zoom))

        # Create thick slices
        orig_thick = du.get_thick_slices(orig_data, self.slice_thickness)
        assert orig_thick.max() > 0.8, f"Multi Dataset - orig thick fail, max removed {orig_thick.max()}"
        # Make 4D
        orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))
        assert orig_thick.max() > 0.8, "Multi Dataset - transpose fail, max removed  maximum {} minimum {}".format(orig_thick.max(), orig_thick.min())
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transforms

        logger.info(f"Successfully loaded Image from {img_filename}")

    def _get_scale_factor(self):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        scale = self.base_res / np.asarray(self.zoom)

        if self.gn_noise != 0:
            scale += self.gn_noise

        return scale

    def __getitem__(self, index):
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        if self.transforms is not None:
            img = self.transforms(img)

        return {'image': img, 'scale_factor': scale_factor}

    def __len__(self):
        return self.count

# Operator to load hdf5-file for training
class MultiScaleDataset(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, gn_noise=False, transforms=None, distance_transforms=None):

        #assert cfg.DATA.PADDED_SIZE == max(cfg.DATA.SIZES), "The padding size is not equal to max of available sizes"
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.gn_noise = gn_noise

        # Load the h5 file and save it to the datase
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode
        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dset = list(hf[f'{size}']['orig_dataset'])
                    logger.info("Processed origs of size {} in {:.3f} seconds".format(size, time.time()-start))
                    self.images.extend(img_dset)
                    self.labels.extend(list(hf[f'{size}']['aseg_dataset']))
                    logger.info("Processed asegs of size {} in {:.3f} seconds".format(size, time.time()-start))
                    self.weights.extend(list(hf[f'{size}']['weight_dataset']))
                    self.zooms.extend(list(hf[f'{size}']['zoom_dataset']))
                    logger.info("Processed zooms of size {} in {:.3f} seconds".format(size, time.time() - start))
                    logger.info("Processed weights of size {} in {:.3f} seconds".format(size, time.time()-start))
                    self.subjects.extend(list(hf[f'{size}']['subject']))
                    logger.info("Processed subjects of size {} in {:.3f} seconds".format(size, time.time()-start))
                    logger.info(f"Number of slices for size {size} is {len(img_dset)}")

                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue

            self.count = len(self.images)
            self.transforms = transforms

            # distance field transformations
            self.distancetransform = distance_transforms

            logger.info("Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(self.count, dataset_path, cfg.DATA.PLANE, time.time()-start))
            # except Exception as e:
            #     print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img_zoom, scale_aug):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        if torch.all(scale_aug > 0):
            img_zoom *= (1 / scale_aug)

        scale = self.base_res / img_zoom

        if self.gn_noise:
            scale += torch.randn(1) * 0.1 + 0 # needs to be changed to torch.tensor stuff
            scale = torch.clamp(scale, min=0.1)

        return scale

    def _pad(self, image):

        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros((self.max_size, self.max_size), dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros((self.max_size, self.max_size, c), dtype=image.dtype)

        if self.max_size < h:
            sub = h - self.max_size
            padded_img = image[0: h - sub, 0: w - sub]
        else:
            padded_img[0: h, 0: w] = image

        return padded_img

    def unify_imgs(self, img, label, weight):

        img = self._pad(img)
        label = self._pad(label)
        weight = self._pad(weight)

        return img, label, weight

    def __getitem__(self, index):

        padded_img, padded_label, padded_weight = self.unify_imgs(self.images[index], self.labels[index], self.weights[index])
        img = np.expand_dims(padded_img.transpose((2, 0, 1)), axis=3)
        assert img.max() > 0, "L184 Wrong maximum {} {}".format(img.max(), img.min())
        label = padded_label[np.newaxis, :, :, np.newaxis]
        weight = padded_weight[np.newaxis, :, :, np.newaxis]

        subject = tio.Subject({'img': tio.ScalarImage(tensor=img),
                               'label': tio.LabelMap(tensor=label),
                               'weight': tio.LabelMap(tensor=weight)}
                              )

        zoom_aug = torch.as_tensor([0., 0.])

        if self.transforms is not None:
            tx_sample = self.transforms(subject) # this returns data as torch.tensors

            img = torch.squeeze(tx_sample['img'].data).float() #.astype(np.float32)
            label = torch.squeeze(tx_sample['label'].data).byte() #.astype(np.uint8)
            weight = torch.squeeze(tx_sample['weight'].data).float() #.astype(np.float32)

            # get updated scalefactor, incase of scaling, not ideal - fails if scales is not in dict
            rep_tf = tx_sample.get_composed_history()
            if rep_tf:
                zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])[:-1]

            # Normalize image and clamp between 0 and 1
            img = torch.clamp(img / img.max(), min=0.0, max=1.0)

        if self.distancetransform is not None:
            # If distance transform, weight is this and not ce-weight mask.
            weight = self.distancetransform({'label': label, 'zooms': self.zooms[index]})

        scale_factor = self._get_scale_factor(torch.from_numpy(self.zooms[index]), scale_aug=zoom_aug)
        assert (scale_factor > 0).all(), f"Scale factor has negative values: {scale_factor}, Zooms: {self.zooms[index]}"

        return {'image': img, 'label': label, 'weight': weight,
                "scale_factor": scale_factor}

    def __len__(self):
        return self.count


# Operator to load hdf5-file for training
class MultiScaleDatasetIt(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, gn_noise=False, transforms=None):

        assert cfg.DATA.PADDED_SIZE == max(cfg.DATA.SIZES), "The padding size is not equal to max of available sizes"
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.gn_noise = gn_noise
        self.load_to_memory = False
        self.sizes = cfg.DATA.SIZES

        # Load the h5 file and save it to the datasets
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []
        self._data = {"images": [], "labels": [], "weights": [], "subjects": [], "zooms": []}
        self._params = {"img_name": "orig_dataset", "labels": "aseg_dataset", "weights": "weight_dataset",
                        "zoom": "zoom_dataset",
                        "subject_name": "subject"}

        # Open file in reading mode
        start = time.time()
        try:
            # force conversion to iterable
            try:
                file_iter = iter([dataset_path] if isinstance(dataset_path, str) else dataset_path)
            except TypeError as e:
                raise TypeError("dataset_path has to be a string or an iterable list of strings")

            # open file in reading mode
            for dset_file in file_iter:
                hf = h5py.File(dset_file, "r")
                #with h5py.File(dset_file, "r") as hf: # can not be used because it would close the file
                self._load(hf)

            self.sizes = self._h5length(self._data["images"])
            self.count = np.sum(self.sizes)
            self.cumsizes = np.cumsum(self.sizes)
            # self.count = np.sum(self._h5length(self._data["images"]))
            print(self.count)
            #self.count = len(self.images)
            self.transforms = transforms

            logger.info("Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(self.count,
                                                                                                     dataset_path,
                                                                                                     cfg.DATA.PLANE,
                                                                                                     time.time()-start))
            # except Exception as e:
            #     print("Loading failed: {}".format(e))

        except NotImplementedError as e:
            print(e)

    @staticmethod
    def _h5length(h5_list):
        """Static function to get number of samples from list."""
        return [h5_list[i].shape[0] for i in range(len(h5_list))]

    @staticmethod
    def _init_field(num: int, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            return np.empty_like((num,) + src.shape[1:])
        else:
            return [""] * num

    def _load(self, hf):
        """Iteratively loads files into memory."""

        source_to_target_img = {
            self._params["img_name"]: "images",
            self._params["subject_name"]: "subjects",
            self._params["labels"]: "labels",
            self._params["weights"]: "weights",
            self._params["zoom"]: "zooms"
        }
        self._source_to_target_assign(source_to_target_img, hf)

    def _source_to_target_assign(self, source_to_target, root):
        """Helper function to copy hdf5 data to ram, if required."""
        for s, t in source_to_target.items():
            for size in self.sizes[::-1]:
                try:
                    data = root[f'{size}'][s]
                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue

                if self.load_to_memory:
                    data = np.asarray(data)
                self._data[t].append(data)

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img_zoom, scale_aug):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """

        if torch.all(scale_aug > 0):
            img_zoom *= (1 / scale_aug)

        scale = self.base_res / img_zoom

        if self.gn_noise:
            scale += torch.randn(1) * 0.1 + 0

        return scale

    def _pad(self, image):
        if len(image.shape) == 2:
            h, w = image.shape
            padded_img = np.zeros((self.max_size, self.max_size), dtype=image.dtype)
        else:
            h, w, c = image.shape
            padded_img = np.zeros((self.max_size, self.max_size, c), dtype=image.dtype)

        padded_img[0: h, 0: w] = image
        return padded_img

    def unify_imgs(self, index_slice): #img, label, weight):

        img = self._pad(self._get_sample(index_slice, "images"))
        label = self._pad(self._get_sample(index_slice, "labels"))
        weight = self._pad(self._get_sample(index_slice, "weights"))

        return img, label, weight

    def _get_sample(self, index_slice, key):
        return self._index(self._data[key], index_slice)

    def _index(self, h5_list, index):
        if isinstance(index, int):
            return self._index_list(h5_list, [index])
        elif isinstance(index, list):
            return self._index_list(h5_list, index)
        else:
            raise ValueError("index should be either int or list, but was %s." % type(index).__name__)

    def _index_list(self, h5_list, index):
        #create empty array with dimensions batch, h,w,c
        out = np.empty(h5_list[0].shape[1:])

        # list with dimension of images (256)

        #if index[0] == 0:
            #print("Sizes", self.sizes, h5_list[0].shape, h5_list[1].shape)
            #print("Cumsizes", self.cumsizes)

        # determine which list to query: #208050+6228+7430
        j = 0 if index < self.cumsizes[0] else 1 if index < self.cumsizes[1] else 2
        #if index[0] == 0:
            #print("index", index)
        #print("Index", index)
        index[0] -= self.cumsizes[j-1] if j != 0 else 0
        #if index[0] == 0:
            #print("Index2", index, j, self.cumsizes[j], self.cumsizes[j-1])
        out = self._assign_field(0, index, out, h5_list[j])
        #if index[0] == 0:
            #print("Outshape", out.shape)
        # loop over h5_list (1 for normal arrays)
        """
        for j in range(1): #len(h5_list)):
            select = [(_idx, i) for _idx, i in enumerate(index) if i >= prev and i < cumsizes[j]]
            print("select", select)
            # target new one (0-batch size), source = original array (index1 - index batch)
            t_idx = [i[0] for i in select]
            s_idx = [i[1] for i in select]
            print("idx", t_idx, s_idx)
            # overwrite empty array with newly assigned h5_list entry
            out = self._assign_field(t_idx, s_idx, out, h5_list[j])
            # 256 - 512 - ... (for three D volumes?)
            prev = cumsizes[j]
        """
        return out

    def _assign_field(self, t_index, s_index, out, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            if self._params["load_to_memory"]:
                out[t_index] = src[s_index]
            else:
                out[t_index] = np.asarray(src[s_index]).squeeze()
        else:
            out = src[s_index].squeeze()
        return out

    def __getitem__(self, index):

        padded_img, padded_label, padded_weight = self.unify_imgs(index)
        img = np.expand_dims(padded_img.transpose((2, 0, 1)), axis=3)
        label = padded_label[np.newaxis, :, :, np.newaxis]
        weight = padded_weight[np.newaxis, :, :, np.newaxis]

        subject = tio.Subject({'img': tio.ScalarImage(tensor=img),
                               'label': tio.LabelMap(tensor=label),
                               'weight': tio.LabelMap(tensor=weight)}
                              )

        zoom_aug = torch.as_tensor([0., 0.])

        if self.transforms is not None:
            tx_sample = self.transforms(subject) # this returns data as torch.tensors

            img = torch.squeeze(tx_sample['img'].data).float() #.astype(np.float32)
            label = torch.squeeze(tx_sample['label'].data).byte() #.astype(np.uint8)
            weight = torch.squeeze(tx_sample['weight'].data).float() #.astype(np.float32)

            # get updated scalefactor, incase of scaling, not ideal - fails if scales is not in dict
            rep_tf = tx_sample.get_composed_history()
            if rep_tf:
                zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])[:-1]

            # Normalize image and clamp between 0 and 1
            img = torch.clamp(img / img.max(), min=0.0, max=1.0)

        scale_factor = self._get_scale_factor(torch.from_numpy(self._get_sample(index, "zooms")), scale_aug=zoom_aug)

        return {'image': img, 'label': label, 'weight': weight,
                "scale_factor": scale_factor}

    def __len__(self):
        return self.count

# Operator to load hdf5-file for validation
class MultiScaleDatasetVal(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, transforms=None, distance_transforms=None):

        assert cfg.DATA.PADDED_SIZE == max(cfg.DATA.SIZES), "The padding size is not equal to max of available sizes"
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES

        # Load the h5 file and save it to the dataset
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []

        # Open file in reading mode
        start = time.time()
        with h5py.File(dataset_path, "r") as hf:
            for size in cfg.DATA.SIZES:
                try:
                    logger.info(f"Processing images of size {size}.")
                    img_dset = list(hf[f'{size}']['orig_dataset'])
                    logger.info("Processed origs of size {} in {:.3f} seconds".format(size, time.time() - start))
                    self.images.extend(img_dset)
                    self.labels.extend(list(hf[f'{size}']['aseg_dataset']))
                    logger.info("Processed asegs of size {} in {:.3f} seconds".format(size, time.time() - start))
                    self.weights.extend(list(hf[f'{size}']['weight_dataset']))
                    logger.info("Processed weights of size {} in {:.3f} seconds".format(size, time.time() - start))
                    self.zooms.extend(list(hf[f'{size}']['zoom_dataset']))
                    logger.info("Processed zooms of size {} in {:.3f} seconds".format(size, time.time() - start))
                    self.subjects.extend(list(hf[f'{size}']['subject']))
                    logger.info("Processed subjects of size {} in {:.3f} seconds".format(size, time.time() - start))
                    logger.info(f"Number of slices for size {size} is {len(img_dset)}")

                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue

        self.count = len(self.images)
        self.transforms = transforms
        logger.info("Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(self.count, dataset_path, cfg.DATA.PLANE, time.time()-start))

        # except Exception as e:
        #     print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img_zoom):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        scale = self.base_res / img_zoom
        return scale

    def __getitem__(self, index):

        img = self.images[index]
        assert img.max() > 0, "L556 Wrong maximum {} {}".format(img.max(), img.min())
        label = self.labels[index]
        weight = self.weights[index]

        scale_factor = self._get_scale_factor(self.zooms[index])
        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight, 'scale_factor': scale_factor})

            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']
            scale_factor = tx_sample['scale_factor']

        if self.distancetransform is not None:
            # If distance transform, weight is this and not ce-weight mask.
            weight = self.distancetransform({'label': label, 'zooms': self.zooms[index]})
        assert (scale_factor > 0).all(), f"Scale factor has negative values: {scale_factor}, Zooms: {self.zooms[index]}"
        return {'image': img, 'label': label, 'weight': weight,
                'scale_factor': scale_factor}

    def __len__(self):
        return self.count


# Operator to load hdf5-file for validation
class MultiScaleDatasetValIt(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, transforms=None, distance_transforms=None):

        assert cfg.DATA.PADDED_SIZE == max(cfg.DATA.SIZES), "The padding size is not equal to max of available sizes"
        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.load_to_memory =False
        self.sizes = cfg.DATA.SIZES

        # Load the h5 file and save it to the dataset
        self.images = []
        self.labels = []
        self.weights = []
        self.subjects = []
        self.zooms = []
        self._data = {"images": [], "labels": [], "weights": [], "subjects": [], "zooms": []}
        self._params = {"img_name": "orig_dataset", "labels": "aseg_dataset", "weights": "weight_dataset", "zoom": "zoom_dataset",
                        "subject_name": "subject"}

        # Open file in reading mode
        start = time.time()
        try:
            # force conversion to iterable
            try:
                file_iter = iter([dataset_path] if isinstance(dataset_path, str) else dataset_path)
            except TypeError as e:
                raise TypeError("dataset_path has to be a string or an iterable list of strings")
            #file_iter = dataset_path
            # open file in reading mode
            for dset_file in file_iter:
                hf = h5py.File(dset_file, "r")
                #with h5py.File(dset_file, "r") as hf:
                self._load(hf)

            #self.count = len(self.images)
            #print(self._data["images"])
            #self.count = np.sum(self._h5length(self._data["images"])) #208050+6228+7430
            self.sizes = self._h5length(self._data["images"])
            self.count = np.sum(self.sizes)
            self.cumsizes = np.cumsum(self.sizes)
            #self.count = np.sum(self._h5length(self._data["images"]))
            print(self.count)
            self.transforms = transforms

            # distance field transformations
            self.distancetransform = distance_transforms

            logger.info("Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(self.count, dataset_path, cfg.DATA.PLANE, time.time()-start))
            print(
                "Successfully loaded {} data from {} with plane {} in {:.3f} seconds".format(self.count, dataset_path,
                                                                                             cfg.DATA.PLANE,
                                                                                             time.time() - start))

        # except Exception as e:
        #     print("Loading failed: {}".format(e))
        except NotImplementedError as e:
            print(e)

    @staticmethod
    def _h5length(h5_list):
        """Static function to get number of samples from list."""
        return [h5_list[i].shape[0] for i in range(len(h5_list))]

    @staticmethod
    def _init_field(num: int, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            return np.empty_like((num,) + src.shape[1:])
        else:
            return [""] * num

    def _load(self, hf):
        """Iteratively loads files into memory."""

        source_to_target_img = {
            self._params["img_name"]: "images",
            self._params["subject_name"]: "subjects",
            self._params["labels"]: "labels",
            self._params["weights"]: "weights",
            self._params["zoom"]: "zooms"
        }
        self._source_to_target_assign(source_to_target_img, hf)

    def _source_to_target_assign(self, source_to_target, root):
        """Helper function to copy hdf5 data to ram, if required."""
        for s, t in source_to_target.items():
            for size in self.sizes[::-1]:
                try:
                    data = root[f'{size}'][s]
                except KeyError as e:
                    print(f"KeyError: Unable to open object (object {size} does not exist)")
                    continue

                if self.load_to_memory:
                    data = np.asarray(data)
                self._data[t].append(data)
                #self._data[t].extend(data) --> loads data into memory

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img_zoom):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        if np.all(img_zoom != 1.0) and np.all(img_zoom != 0.8) and np.all(img_zoom != 0.7):
            img_zoom += 1
        scale = self.base_res / img_zoom

        return scale

    def _get_sample(self, index_slice, key):
        return self._index(self._data[key], index_slice)

    def _index(self, h5_list, index):
        if isinstance(index, int):
            return self._index_list(h5_list, [index])
        elif isinstance(index, list):
            return self._index_list(h5_list, index)
        else:
            raise ValueError("index should be either int or list, but was %s." % type(index).__name__)


    def _index_list(self, h5_list, index):
        #create empty array with dimensions batch, h,w,c
        out = np.empty(h5_list[0].shape[1:])

        # list with dimension of images (256)

        #if index[0] == 0:
            #print("Sizes", self.sizes, h5_list[0].shape, h5_list[1].shape)
            #print("Cumsizes", self.cumsizes)

        # determine which list to query: #208050+6228+7430
        j = 0 if index < self.cumsizes[0] else 1 if index < self.cumsizes[1] else 2
        #if index[0] == 0:
            #print("index", index)
        #print("Index", index)
        index[0] -= self.cumsizes[j-1] if j != 0 else 0
        #if index[0] == 0:
            #print("Index2", index, j, self.cumsizes[j], self.cumsizes[j-1])
        out = self._assign_field(0, index, out, h5_list[j])
        #if index[0] == 0:
            #print("Outshape", out.shape)
        # loop over h5_list (1 for normal arrays)
        """
        for j in range(1): #len(h5_list)):
            select = [(_idx, i) for _idx, i in enumerate(index) if i >= prev and i < cumsizes[j]]
            print("select", select)
            # target new one (0-batch size), source = original array (index1 - index batch)
            t_idx = [i[0] for i in select]
            s_idx = [i[1] for i in select]
            print("idx", t_idx, s_idx)
            # overwrite empty array with newly assigned h5_list entry
            out = self._assign_field(t_idx, s_idx, out, h5_list[j])
            # 256 - 512 - ... (for three D volumes?)
            prev = cumsizes[j]
        """
        return out

    def _assign_field(self, t_index, s_index, out, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            if self._params["load_to_memory"]:
                out[t_index] = src[s_index]
            else:
                out[t_index] = np.asarray(src[s_index]).squeeze()
        else:
            out = src[s_index].squeeze()
        return out

    def __getitem__(self, index):
        img = self._get_sample(index, "images") # self.images[index]
        label = self._get_sample(index, "labels") # self.labels[index]
        weight = self._get_sample(index, "weights") # self.weights[index]
        scale_factor = self._get_scale_factor(self._get_sample(index, "zooms")) #data["zooms"])#self._get_sample(index, "zooms"))

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight, 'scale_factor': scale_factor})

            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']
            scale_factor = tx_sample['scale_factor']

        return {'image': img, 'label': label, 'weight': weight,
                'scale_factor': scale_factor}

    def __len__(self):
        return self.count


# Operator to load hdf5-file for training
class AsegDatasetWithAugmentation(Dataset):
    """
    Class for loading aseg file with augmentations (transforms)
    """
    def __init__(self, dataset_path, cfg, transforms=None):

        assert cfg.DATA.PADDED_SIZE == max(cfg.DATA.SIZES), "The padding size is not equal to max of available sizes"
        self.max_size = cfg.DATA.PADDED_SIZE
        self.model_height = cfg.MODEL.HEIGHT
        self.model_width = cfg.MODEL.WIDTH

        # Load the h5 file and save it to the dataset
        try:

            # Open file in reading mode
            start = time.time()
            with h5py.File(dataset_path, "r") as hf:
                self.images = np.array(hf.get('orig_dataset'))
                logger.info("Processed origs in {:.3f} seconds".format(time.time() - start))
                self.labels = np.array(hf.get('aseg_dataset'))
                logger.info("Processed asegs in {:.3f} seconds".format(time.time() - start))
                self.weights = np.array(hf.get('weight_dataset'))
                logger.info("Processed weights in {:.3f} seconds".format(time.time() - start))
                self.subjects = np.array(hf.get("subject"))
                logger.info("Processed subjects in {:.3f} seconds".format(time.time() - start))

            self.count = self.images.shape[0]
            self.transforms = transforms

            print("Successfully loaded {} with plane: {} in {:.3f} seconds".format(dataset_path, cfg.DATA.PLANE, time.time()-start))

        except Exception as e:
            print("Loading failed: {}".format(e))

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img):
        h, w, _ = img.shape
        scale_h = float(h) / self.model_height
        scale_w = float(w) / self.model_width

        return np.array([scale_h, scale_w])

    def __getitem__(self, index):

        img = self.images[index]
        label = self.labels[index]
        weight = self.weights[index]
        scale_factor = self._get_scale_factor(img)

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight})
            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']

        return {'image': img, 'label': label, 'weight': weight, "scale_factor": scale_factor}

    def __len__(self):
        return self.count