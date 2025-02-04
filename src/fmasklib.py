# pylint: disable=line-too-long
import os
import sys
from pathlib import Path
from typing import Union
import pandas as pd
from satellite import Landsat, Sentinel2
from phylib import flood_fill_shadow
from sklearn.metrics import accuracy_score, precision_score, recall_score
import predictor as P
import utils
from unetlib import UNet
from rflib import RandomForest, Dataset as PixeDataset
from lightgbmlib import LightGBM
import constant as C
from phylib import Physical, segment_cloud_objects
from copy import deepcopy
from bitlib import BitLayer
import numpy as np
from skimage.filters import threshold_otsu

class Fmask(object):
    """Fmask class
    """

    # %% Attribues
    version = 5.0  # the fmask version
    algorithm = "interaction"  # the algorithm for cloud masking, including "physical", "randomforest", "unet", "interaction"
    image: Union[Landsat, Sentinel2] = (
        None  # the image object can be hold either a Landsat or Sentinel2 object
    )
    physical: Physical = None  # the physical object
    rf_cloud: RandomForest = None  # the random forest model for cloud masking
    lightgbm_cloud: LightGBM = None  # the lightgbm model for cloud masking
    unet_cloud: UNet = None  # the unet model for cloud masking
    unet_shadow: UNet = None  # the unet model for shadow masking
    database_pixel: PixeDataset = (
        None  # the dataset for training the random forest model
    )
    pixelbase: PixeDataset = None  # the dataset for the random forest model
    patchbase = None  # the dataset for the UNet model
    database_patch = None  # the dataset for UNet model

    path = None
    dir_patch = None  # directory of patches for training the unet model
    dir_pixel = None  # directory of patches for training the random forest

    # predictor_full = None  # full predictors that are provided by the module

    resolution = 0  # spatial resolution that we processing the image
    erosion_radius = 0  # the radius of erosion, unit: pixels

    buffer_cloud = 0  # buffer size of cloud in pixels
    buffer_shadow = 3  # buffer size of shadow in pixels
    buffer_snow = 0  # buffer size of snow in pixels

    # define the classes of the cloud and non-cloud, and filled pixels for the machine learning model
    # the pixel value will rely on the index of the defined classes
    cloud_model_classes = ["noncloud", "cloud", "filled"]
    shadow_model_classes = ["nonshadow", "shadow", "filled"]

    base_machine_learning = ["unet"]  # the base machine learning model for cloud masking, such as 'randomforest', 'unet', 'lightgbm', 'lightgbm_unet'
    tune_machine_learning = "lightgbm"  # the machine learning model for tuning the cloud masking, such as 'randomforest', 'unet', 'lightgbm'
    tune_strategy = "transfer"  # 'transfer' or 'new'
    tune_seed = "physical"  # for random forest only ['disagree', 'physical']

    max_iteration = (
        1  # maximum iteration numbers between fmask and machine learning model
    )
    disagree_rate = 0.25  # the rate of disagreement between two consective iteration by machine learning
    seed_levels = [0, 0] # percentile of selecting non-cloud seeds and cloud seeds
 
    physical_rules_dynamic = True  # able to change rules during the iterations
    # sets of other
    show_figure = False  # indicates show the figures during the progress or not

    # inner variables
    _machine_learn_models = ["randomforest", "lightgbm", "unet"]
    # the valid pixel values in reference data
    _valid_class_labels = [
        C.LABEL_CLEAR,
        C.LABEL_WATER,
        C.LABEL_LAND,
        C.LABEL_SNOW,
        C.LABEL_SHADOW,
        C.LABEL_CLOUD,
        C.LABEL_FILL,
    ]

    @property
    def valid_class_labels(self):
        """
        Returns the valid class labels for the Fmask algorithm.

        Returns:
            list: A list of valid class labels.
        """
        return self._valid_class_labels

    # masks
    cloud: BitLayer = None  # the cloud mask
    cloud_region = None  # the cloud region list, will be used in shadow masking
    cloud_object = None  # the cloud object mask, will be used in shadow masking
    shadow = None  # the shadow mask
    probability = None  # the cloud probability layer

    @property
    def full_predictor(self) -> list:
        """
        Returns a list of predictors based on the selected algorithm.

        Returns:
            list: A list of predictors to be used in the model.
        """
        # select the predictors according to the algorithm given
        if self.algorithm == "physical":
            predictors = self.physical.predictors.copy()
        elif self.algorithm == "lightgbm":
            predictors = self.lightgbm_cloud.predictors.copy()
            predictors = predictors + self.physical.predictors # no matter what, we need to use the physical predictors to create variables to match shadows
        elif self.algorithm == "randomforest":
            predictors = self.rf_cloud.predictors.copy()
            predictors = predictors + self.physical.predictors
        elif self.algorithm == "unet":
            predictors = self.unet_cloud.predictors.copy()
            predictors = predictors + self.physical.predictors
        elif self.algorithm == "interaction":
            predictors = self.physical.predictors.copy()
            if ("unet" in self.base_machine_learning) | (self.tune_machine_learning == "unet"):
                predictors = predictors + self.unet_cloud.predictors
            if ("randomforest" in self.base_machine_learning) | (self.tune_machine_learning == "randomforest"):
                predictors = predictors + self.rf_cloud.predictors
            if ("lightgbm" in self.base_machine_learning) | (self.tune_machine_learning == "lightgbm"):
                predictors = predictors + self.lightgbm_cloud.predictors
        return list(set(predictors)) # unique the predictors

    @property
    def ensemble_mask(self):
        """
        Generates an ensemble mask based on different classification results.

        Returns:
            numpy.ndarray: The ensemble mask with labeled regions for water, snow, shadow, cloud, and fill.
        """
        mask = np.zeros(self.image.obsmask.shape, dtype="uint8")
        if self.physical is not None:
            mask[self.physical.water] = C.LABEL_WATER
            mask[self.physical.snow] = C.LABEL_SNOW
        if self.shadow is not None:
            if self.buffer_shadow > 0:
                mask[utils.dilate(self.shadow)] = C.LABEL_SHADOW
            else:
                mask[self.shadow] = C.LABEL_SHADOW
        if self.cloud is not None:
            if self.buffer_cloud > 0:
                mask[utils.dilate(self.cloud.last)] = C.LABEL_CLOUD
            else:
                mask[self.cloud.last] = C.LABEL_CLOUD
        mask[self.image.filled] = C.LABEL_FILL
        return mask

    @property
    def cloud_percentage(self):
        """
        Returns the percentage of cloud coverage in the image.

        Returns:
            float: The percentage of cloud coverage.
        """
        return np.count_nonzero(np.bitwise_and(self.cloud.last, self.image.obsmask)) / np.count_nonzero(self.image.obsmask)

    def set_base_machine_learning(self, models: str) -> None:
        """
        Sets the base machine learning model for the cloud masking algorithm.

        Args:
            models (str): The base machine learning model to set, with each model separated by an underscore.
        """
        self.base_machine_learning = models.split("_")

    def set_tune_machine_learning(self, model: str) -> None:
        """
        Sets the tune machine learning model for the cloud masking algorithm.

        Args:
            models (str): The base machine learning model to set, with each model separated by an underscore.
        """
        self.tune_machine_learning = model

    def get_patch_data_index(self, predictors):
        """
        Get the index of the predictors that are used in the patch dataset.

        Returns:
            list: The index of the predictors.
        """
        # see create_train_data_patch.py for the predictors used in the patch dataset generation process
        return [
            i for i, pre in enumerate(P.l8_predictor_cloud_cnn) if pre in predictors
        ]

    # %% Methods
    def init_modules(self) -> None:
        """
        nitialize and optimize the cloud models based on the spacecraft type.

        This method initializes and configures the cloud models based on the spacecraft type.
        It sets the appropriate parameters and values for each model.

        Returns:
            None
        """
        # initialize the cloud models without initialization according to the spacecraft
        spacecraft = self.image.spacecraft.upper()
        if spacecraft in ["LANDSAT_8", "LANDSAT_9"]:
            self.physical = Physical(
                predictors=P.l8_predictor_cloud_phy.copy(), woc=0.3, threshold=0.175, overlap=0.0
            )
            self.rf_cloud = RandomForest(
                classes=["noncloud", "cloud"],
                predictors=P.l8_predictor_cloud_pixel.copy(),
                nsamples=10000,
                ntrees=100,
                tune_update_rate=0.05,
                path=os.path.join(self.dir_package, "model", "rf_nc_l8.pk"),
            )
            self.lightgbm_cloud = LightGBM(
                classes=["noncloud", "cloud"],
                num_leaves=25,
                min_data_in_leaf=500,
                tune_update_rate=0.05,
                predictors=P.l8_predictor_cloud_pixel.copy(),
                path=os.path.join(self.dir_package, "model", "lightgbm_nc_l8.pk"),
            )
            self.unet_cloud = UNet(
                classes=["noncloud", "cloud", "filled"],
                predictors=P.l8_predictor_cloud_cnn.copy(),
                learn_rate=1e-3,
                epoch=60,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                tune_epoch=10,
                path=os.path.join(self.dir_package, "model", "unet_ncf_l8.pt"),
            )
            self.unet_shadow = UNet(
                classes=["nonshadow", "shadow", "filled"],
                predictors=P.l8_predictor_shadow_cnn.copy(),
                learn_rate=1e-3,
                epoch=80,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                path=None,
            )
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = int(90/self.resolution)  # the radius of erosion, unit: pixels

        elif spacecraft in ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7"]:
            self.physical = Physical(
                predictors=P.l7_predictor_cloud_phy.copy(), woc=0.0, threshold=0.1, overlap=0.0
            )
            self.rf_cloud = RandomForest(
                classes=["noncloud", "cloud"],
                predictors=P.l7_predictor_cloud_pixel.copy(),
                nsamples=10000,
                ntrees=100,
                tune_update_rate=0.05,
                path=os.path.join(self.dir_package, "model", "rf_nc_l7.pk"),
            )
            self.lightgbm_cloud = LightGBM(
                classes=["noncloud", "cloud"],
                num_leaves=30,
                min_data_in_leaf=300,
                tune_update_rate=0.05,
                predictors=P.l7_predictor_cloud_pixel.copy(),
                path=os.path.join(self.dir_package, "model", "lightgbm_nc_l7.pk"),
            )
            self.unet_cloud = UNet(
                classes=["noncloud", "cloud", "filled"],
                predictors=P.l7_predictor_cloud_cnn.copy(),
                learn_rate=1e-3,
                epoch=60,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                tune_epoch=10,
                path= os.path.join(self.dir_package, "model", "unet_ncf_l7.pt"),
            )
            self.unet_shadow = UNet(
                classes=["nonshadow", "shadow", "filled"],
                predictors=P.l7_predictor_shadow_cnn.copy(),
                learn_rate=1e-3,
                epoch=80,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                path=None,
            )
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = int(150/self.resolution)  # the radius of erosion, unit: pixels
        elif spacecraft in ["SENTINEL-2A", "SENTINEL-2B", "SENTINEL-2C"]:
            self.physical = Physical(
                predictors=P.s2_predictor_cloud_phy.copy(), woc=0.5, threshold=0.2, overlap=0.0
            )

            self.rf_cloud = RandomForest(
                classes=["noncloud", "cloud"],
                predictors=P.s2_predictor_cloud_pixel.copy(),
                nsamples=10000,
                ntrees=100,
                tune_update_rate=0.05,
                path=os.path.join(self.dir_package, "model", "rf_nc_s2.pk"),
            )

            self.lightgbm_cloud = LightGBM(
                classes=["noncloud", "cloud"],
                num_leaves=10,
                min_data_in_leaf=200,
                tune_update_rate=0.05,
                predictors=P.s2_predictor_cloud_pixel.copy(),
                path=os.path.join(self.dir_package, "model", "lightgbm_nc_s2.pk"),
            )

            self.unet_cloud = UNet(
                classes=["noncloud", "cloud", "filled"],
                predictors=P.s2_predictor_cloud_cnn.copy(),
                learn_rate=1e-3,
                epoch=60,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                tune_epoch=10,
                path=os.path.join(self.dir_package, "model", "unet_ncf_s2.pt"),
            )

            self.unet_shadow = UNet(
                classes=["nonshadow", "shadow", "filled"],
                predictors=P.s2_predictor_shadow_cnn.copy(),
                learn_rate=1e-3,
                epoch=80,
                patch_size=512,
                patch_stride_train=488,
                patch_stride_classify=488,
                path=None,
            )
            self.resolution = 20  # spatial resolution that we processing the image
            self.erosion_radius = int(90/self.resolution)   # the radius of erosion, unit: pixels

    def init_pixelbase(
        self,
        directory=None,
        datasets=None,
        classes=None,
        sampling_methods=None,
        number=10000,
        exclude=None,
    ) -> None:
        """initialize the dataset for training the random forest model

        Args:
            directory (str): The directory to save the dataset.
            datasets (list): The datasets to use for collecting training data.
            classes (list): The classes of the dataset.
            sampling_methods (list): The sampling methods of the dataset.
            number (int): The number of samples to collect.
            exclude (str): Image will be excluded from the training data.

        """
        # locate the pixel datasets
        spacecraft = self.image.spacecraft.upper()
        if spacecraft.startswith("L"):
            if directory is None:
                directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Landsat8"
                datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD"]
        elif spacecraft.startswith("S"):
            if directory is None:
                directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Sentinel2"
                datasets = ["S2ALCD", "S2WHUCDPLUS", "S2IRIS", "S2FMASK"]
        if classes is None:
            classes = self.rf_cloud.classes
        if sampling_methods is None:
            sampling_methods = ["stratified", "stratified"]
        # init the pixelbase
        self.pixelbase = PixeDataset(
            directory,
            datasets,
            classes,
            sampling_methods,
            number=number,
            exclude=exclude,
        )
        # forward the dataset to the random forest model
        self.rf_cloud.sample = self.pixelbase
        self.lightgbm_cloud.sample = self.pixelbase

    def load_image(self) -> None:
        """Load image according to the configuration and forward image to the models

        This method loads the image using the specific bands that are known.
        It then forwards the dataset to the models for further processing.

        Args:
            None

        Returns:
            None
        """
        # load image with the specific bands that we know
        self.image.load_data(self.full_predictor)

        # forward dataset to the models
        self.physical.image = self.image
        self.lightgbm_cloud.image = self.image
        self.rf_cloud.image = self.image
        self.unet_cloud.image = self.image
        self.unet_shadow.image = self.image

    def generate_train_data_pixel(
        self, dataset, number, destination=None
    ) -> pd.DataFrame:
        """Collects training data for pixels.

        This method collects training data for pixels based on the specified dataset and number of samples.
        It saves the collected training data as a CSV file if a path is provided.

        Args:
            dataset (str): The dataset to use for collecting training data.
            path (str): The path to save the collected training data. (default: None)
            number (int): The number of samples to collect.

        Returns:
            pf_sample (Dataframe): The collected training data as a pandas DataFrame.
        """
        landcover = self.image.load_landcover()
        reference = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=self.image.shape
        )

        # check the reference mask's classes
        labels_in = np.unique(reference)
        print(f">>> unique values in the reference mask {labels_in}")
        labels_invalid = labels_in[
            [lab not in self._valid_class_labels for lab in labels_in]
        ]
        if len(labels_invalid) > 0:
            print(
                f">>> the reference mask is not in the valid classes {labels_invalid}"
            )
            sys.exit(0)  # exit the program if error occurs

        # start to collect the samples
        pf_sample = utils.collect_sample_pixel(
            self.image.data.data,
            self.image.data.bands,
            reference,
            landcover=landcover,
            number=number,
        )
        print(f">>> {len(pf_sample):012d} samples have been collected")
        # create the directory if it does not exist
        if destination is not None:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            pf_sample.to_csv(destination)
            print(f">>> saved to {destination}")
        return pf_sample

    def generate_train_data_patch(
        self, dataset, path, dformat="tif", append_end=True
    ) -> None:
        """
        Generate training data patches.

        Args:
            dataset (str): The dataset to generate patches for.
            path (str): The path to save the generated patches.
            append_end (bool, optional): Whether to append patches to the end of the existing dataset. Defaults to True.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        reference = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=self.image.shape
        )

        # check the reference mask's classes
        labels_in = np.unique(reference)
        print(f">>> unique values in the reference mask {labels_in}")
        labels_invalid = labels_in[
            [lab not in self._valid_class_labels for lab in labels_in]
        ]
        if len(labels_invalid) > 0:
            print(
                f">>> the reference mask is not in the valid classes {labels_invalid}"
            )
            sys.exit(0)  # exit the program if error occurs

        # start to collect the samples
        utils.collect_sample_patch(
            dataset,
            self.image.name,
            self.image.profile,
            self.image.profilefull,
            self.image.data,
            reference,
            path,
            size=self.unet_cloud.patch_size,  # the size of the patch
            stride=self.unet_cloud.patch_stride_train,  # the stride of the patch
            append_end=append_end,
            dformat=dformat,
        )

    def mask_cloud_pcp(self):
        """
        Masks the pixel cloud probability (pcp) and assigns the result to the `cloud` attribute.
        Also assigns the absolute clear probability (abs_clear) to the `shadow` attribute.
        """
        # init cloud layer
        if self.physical.activated is None:
            self.physical.init_cloud_probability()
        self.cloud = BitLayer(self.image.shape)
        self.cloud.append(self.physical.pcp)

    def mask_shadow(self, potential = "flood"):
        """parent function to mask the shadow"""
        if self.physical.activated: # only when the physical model is activated
            self.mask_shadow_geometry(potential = potential)
        else: # only when the physical model is not activated, we will use non-PCP pixels as shadow because of the extremely large clouds in the imagery
            self.mask_shadow_pcp()
            
    def mask_shadow_pcp(self):
        """
        Masks the pixel cloud probability (pcp) and assigns the result to the `cloud` attribute.
        Also assigns the absolute clear probability (abs_clear) to the `shadow` attribute.
        """
        self.shadow = self.physical.abs_clear

    def mask_cloud_physical(self):
        """mask clouds by the default physical rules"""
        # init cloud layer
        self.cloud = BitLayer(self.image.shape)

        if self.physical.activated is None:
            self.physical.init_cloud_probability()

        if self.physical.activated:
            _options = self.physical.options.copy()  # save the original options
            self.physical.options = [
                True,
                True,
                True,
            ]  # make sure all the physical rules are activated at default
            
            cloud_ph = self.physical.cloud
            # also get the extremely cold clouds when thermal band is available
            cold_cloud = self.physical.cold_cloud
            if cold_cloud is not None:
                cloud_ph[cold_cloud] = 1
            self.cloud.append(self.physical.cloud)
            # show cloud probabilities
            if self.show_figure:
                # show the cloud probabilities
                if self.physical.options[0]:
                    utils.show_cloud_probability(
                        self.physical.prob_variation,
                        self.physical.image.filled,
                        "Spectral variation probability",
                    )
                if self.physical.options[1]:
                    utils.show_cloud_probability(
                        self.physical.prob_temperature,
                        self.physical.image.filled,
                        "Temperature probability",
                    )
                if self.physical.options[2]:
                    utils.show_cloud_probability(
                        self.physical.prob_cirrus,
                        self.physical.image.filled,
                        "Cirrus probability",
                    )
                # make title with the options at the end (TTT)
                utils.show_cloud_probability(
                    self.physical.prob_cloud,
                    self.physical.image.filled,
                    f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})",
                )
            self.physical.options = _options.copy()  # recover the original options
            del _options  # empty the variable that will not be used anymore
        else:  # get the pcp
            self.cloud.append(self.physical.pcp)
        # show cloud mask
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(
                cloud_mask, self.cloud_model_classes, "Physical rules"
            )

    # post processing and generate cloud objects
    def create_cloud_object(self, min_area = 3, postprocess = "none"):
        """
        Erodes the false positive cloud pixels.
        Returns:
            None
        """
        if (postprocess == "none"):
            # no postprocessing
            [cloud_objects, cloud_regions] = segment_cloud_objects(self.cloud.last.copy(), min_area=min_area, buffer2connect=0)
        elif (postprocess == "morphology"):
            # only when the physical model is activated
            if not self.physical.activated:
                return
            # morphology-based, follow Qiu et al., 2019 RSE
            if C.MSG_FULL:
                print(">>> postprocessing with morphology-based elimination")

            # get the potential false positive cloud pixels
            pfpl = self.mask_potential_bright_surface()
            # erode the false positive cloud pixels
            pixels_eroded = utils.erode(self.cloud.last.copy(), radius = self.erosion_radius)
            pfpl = (~pixels_eroded) & pfpl & (~self.physical.water) & self.image.obsmask # indicate the eroded cloud pixels over land
            del pixels_eroded

            # segment the cloud pixels into objects, and if all the pixels of cloud are over the eroded layer, then remove the cloud object
            # remove the small cloud objects less than 3 pixels
            [cloud_objects, cloud_regions] = segment_cloud_objects(self.cloud.last.copy(), min_area=min_area, buffer2connect=0, exclude=pfpl, exclude_method = 'all')
            del pfpl
            
            # remove the small cloud objects 
            if self.image.data.exist("cdi"):
                # Pre-calculate the mask for `_cdi > -0.5
                cdi_mask = self.image.data.get("cdi") > -0.5
                false_small_cloud = np.zeros_like(cloud_objects, dtype=bool) # initialize the cloud_objects as 0
                valid_indices = []
                for icloud, cld_obj in enumerate(cloud_regions):
                    # if (cld_obj.area < 10000) and (np.min(_cdi[cld_obj.coords[:, 0], cld_obj.coords[:, 1]]) > -0.5): # minimum cdi value is larger than -0.5
                    if (cld_obj.area < 10000) and (np.all(cdi_mask[cld_obj.coords[:, 0], cld_obj.coords[:, 1]])): # Check if all cdi values are > -0.5
                        false_small_cloud[cld_obj.coords[:, 0], cld_obj.coords[:, 1]] = True
                    else:
                        valid_indices.append(icloud)      
                cloud_objects[false_small_cloud] = 0
                del false_small_cloud, cdi_mask
                cloud_regions = [cloud_regions[i] for i in valid_indices]
                del valid_indices
            
        elif (postprocess == "unet"):
            # only when the physical model is activated
            if not self.physical.activated:
                return
            if C.MSG_FULL:
                print(">>> postprocessing with UNet-based elimination")
            [cloud_objects, cloud_regions] = segment_cloud_objects(self.cloud.last.copy(), min_area=min_area, buffer2connect=0, exclude=(self.cloud.first==0), exclude_method = 'all')

        # assign the cloud objects and regions 
        self.cloud_object = cloud_objects
        self.cloud_region = cloud_regions
        # update the cloud mask
        self.cloud.append(self.cloud_object >0)
        

    def mask_potential_bright_surface(self):
        """
        Masks the potential bright surface pixels in the image.
        Returns:
            numpy.ndarray: The mask indicating the potential bright surface pixels.
        """
        
        # over urban by use the ndbi
        ndbi = self.image.data.get("ndbi")
        ndbi = utils.enhance_line(ndbi)
        # urban pixels
        pfpl = (ndbi > 0) & (ndbi > self.image.data.get("ndvi")) & (~self.physical.water)
        
        if np.any(pfpl): # only when the potential false positive cloud pixels are available
            # ostu threshold to exclude cloud over the layer if the thermal band is available
            if self.image.data.exist("tirs1"):
                # exclude the extremely cold pixels over the urban pixels
                cold_t = threshold_otsu(self.image.data.get("tirs1")[pfpl])  # Otsu's method
                pfpl[self.image.data.get("tirs1") < cold_t] = 0
            # exclude the confident cloud pixels by cdi
            if self.image.data.exist("cdi"):
                pfpl[self.image.data.get("cdi") < -0.8] = 0  # Follow David, 2018 RSE for Sentinel-2
        
        # Add potential snow/ice pixels in mountain areas
        if self.image.data.exist("slope"):
            _slope = self.image.data.get("slope")
        else:
            _slope = utils.calculate_slope(self.image.data.get("dem"))
        # potential snow/ice pixels in mountain areas
        pfpl = pfpl | ((_slope > 20) & self.physical.snow)

        # Buffer urban pixels with a 500 window to connect the potential false positive cloud pixels into one layer
        radius_pixels = int(250 / self.image.resolution)  # 1 km = 33 Landsat pixels, 500m = 17, 200m = 7
        pfpl = utils.dilate(pfpl, radius=radius_pixels)
        
        # add the snow pixels in normal regions with no dilation in mountain areas
        pfpl = pfpl | self.physical.snow

        return pfpl
            
    def mask_cloud_interaction(self, outcome="classified"):
        """Mask clouds by the interaction of the physical model and machine learning model

        Args:
            outcome (str, optional): Can be "classified" or "physical". Defaults to "classified".
        """
        
        ## Special case: when the update rate is zero or tune_epoch is zero, we do not need to update the model, just use the base model as the base
        # test only, which can be removed in clean version
        if outcome == "classified":
            if self.tune_machine_learning == "unet" and self.unet_cloud.tune_epoch == 0:
                self.mask_cloud_unet()
                return
            elif self.tune_machine_learning == "randomforest" and self.rf_cloud.tune_update_rate == 0:
                self.mask_cloud_random_forest()
                return
            elif self.tune_machine_learning == "lightgbm" and self.lightgbm_cloud.tune_update_rate == 0:
                self.mask_cloud_lightgbm()
                return

        # init cloud layer in bit layers
        self.cloud = BitLayer(self.image.shape)

        #%% init cloud probabilities
        if self.physical.activated is None:
            self.physical.init_cloud_probability()
        # check the physical model activated or not
        if not self.physical.activated:
            print(
                ">>> physical model has not been initialized due to inadquate absolute clear-sky pixels"
            )
            self.cloud.append(self.physical.pcp)
            return

        # display cloud probabilities from the physical rules
        if self.show_figure:
            # show the cloud probabilities
            utils.show_cloud_probability(
                self.physical.prob_variation,
                self.physical.image.filled,
                "Spectral variation",
            )
            utils.show_cloud_probability(
                self.physical.prob_temperature,
                self.physical.image.filled,
                "Temperature/HOT",
            )
            utils.show_cloud_probability(
                self.physical.prob_cirrus,
                self.physical.image.filled,
                "Cirrus",
            )
            # to display the full cloud probability
            self.physical.options = [
                True,
                True,
                True,
            ]
            
            utils.show_cloud_probability(
                self.physical.prob_cloud,
                self.physical.image.filled,
                f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})"
            )

        #%% create initial cloud mask by the machine learning model
        # start to process normal imagery
        # define the class label of the cloud and non-cloud, and filled pixels for the machine learning model
        label_cloud     = self.cloud_model_classes.index("cloud")
        label_noncloud  = self.cloud_model_classes.index("noncloud")
        label_filled    = self.cloud_model_classes.index("filled")

        # load the pretrained unet model if it will be used
        if C.MSG_FULL:
            for mod in self.base_machine_learning:
                print(f">>> loading {mod} as base machine learning model")
            print(f">>> loading {self.tune_machine_learning} as tune machine learning model")
    
        # load the pretrained machine learning models required accordingly
        if (("unet" in self.base_machine_learning) | (self.tune_machine_learning == "unet")) & (not self.unet_cloud.activated):
            self.unet_cloud.load_model()
        if (("randomforest" in self.base_machine_learning) | (self.tune_machine_learning == "randomforest")) & (not self.rf_cloud.activated):
            self.rf_cloud.load_model()
        if (("lightgbm" in self.base_machine_learning) | (self.tune_machine_learning == "lightgbm")) & (not self.lightgbm_cloud.activated):
            self.lightgbm_cloud.load_model()

        # masking initilized clouds
        if C.MSG_FULL:
            for mod in self.base_machine_learning: print(f">>> initilizing cloud mask by {mod}")
        # get the init mask created by the machine learning model
        if len(self.base_machine_learning) == 1:
            # single classifier is used as base machine learning model
            if "unet" in self.base_machine_learning:
                cloud_ml, _ = self.unet_cloud.classify(probability="none")
                cloud_ml[self.image.filled] = label_filled  # exclude the filled pixels by the real extent masking
            elif "randomforest" in self.base_machine_learning:
                cloud_ml, _, subsampling_mask = self.rf_cloud.classify(probability="none", base = True)
                # exclude the filled pixels based on the subsampling mask, which is used to speed up the classification of random forest
                cloud_ml[~subsampling_mask] = label_filled  
            elif "lightgbm" in self.base_machine_learning:
                cloud_ml, _, subsampling_mask = self.lightgbm_cloud.classify(probability="none", base = True)
                # exclude the filled pixels based on the subsampling mask, which is used to speed up the classification of lightgbm
                cloud_ml[~subsampling_mask] = label_filled
        else: # mutiple classifiers are used as base machine learning models
            # initilize variable to store the cloud mask by the base machine learning model
            cloud_ml = None
            if "unet" in self.base_machine_learning:
                cloud_unet, _ = self.unet_cloud.classify(probability="none")
                cloud_unet[self.image.filled] = label_filled  # exclude the filled pixels by the real extent masking
                cloud_ml = cloud_unet
            if "randomforest" in self.base_machine_learning:
                cloud_rf, _, subsampling_mask = self.rf_cloud.classify(probability="none", base = True)
                # exclude the filled pixels based on the subsampling mask, which is used to speed up the classification of random forest
                cloud_rf[~subsampling_mask] = label_filled
                if cloud_ml is None:
                    cloud_ml = cloud_rf
                else:
                    cloud_ml[cloud_rf != cloud_ml] = label_filled # ony remain the agreement pixels
            if "lightgbm" in self.base_machine_learning:
                cloud_lightgbm, _, subsampling_mask = self.lightgbm_cloud.classify(probability="none", base = True)
                # exclude the filled pixels based on the subsampling mask, which is used to speed up the classification of lightgbm
                cloud_lightgbm[~subsampling_mask] = label_filled
                if cloud_ml is None:
                    cloud_ml = cloud_lightgbm
                else:
                    cloud_ml[cloud_lightgbm != cloud_ml] = label_filled

        # record the cloud layer
        if outcome == "classified":
            self.cloud.append(cloud_ml == label_cloud)
        else:
            self.physical.options = [
                True,
                True,
                True,
            ]  # make sure all the physical rules are activated at default
            self.cloud.append(
                self.physical.cloud
            )  # get the default physical cloud mask

        if self.show_figure:
            
            utils.show_cloud_mask(
                cloud_ml, self.cloud_model_classes, "base: " + "&".join(self.base_machine_learning)
            )
            # utils.show_cloud_probability(
            #    prob_ml, self.image.filled, f"base: {self.base_machine_learning_string}"
            # )

        #%% iteract only when the cloud and non-cloud are both in the cloud_ml
        count_cloud = np.count_nonzero(cloud_ml == label_cloud)
        count_noncloud = np.count_nonzero(cloud_ml == label_noncloud)
        # when the pixel-based classifiers are used, we can consider the subsampling sizen to speed up the classification
        if (("randomforest" in self.base_machine_learning) or ("lightgbm" in self.base_machine_learning)):
            # one pixel represent the subsampling_size * subsampling_size pixels after using subsampling to speed up the classification of random forest
            # back to the original number of pixels before subsampling
            ratio_subsampling = self.image.obsnum/np.count_nonzero(cloud_ml != label_filled)
            subsampling_mask = None # only when the first iteration, we do not need to use the subsampling mask because it was based on the random pixels selected for triggering the physical model
            count_cloud = count_cloud*ratio_subsampling
            count_noncloud = count_noncloud*ratio_subsampling
            del ratio_subsampling

        # both cloud and non-cloud are represented enough in the cloud_ml
        if (count_cloud >= self.physical.min_clear) & (count_noncloud >= self.physical.min_clear):
        # if (label_cloud in cloud_ml) & (label_noncloud in cloud_ml):
            # count the labels of cloud and non-cloud
            # self-learning progress
            for i in range(1, self.max_iteration + 1):
                if C.MSG_FULL:
                    print(
                        f">>> adjusting physical rules {i:02d}/{self.max_iteration:02d}"
                    )

                # select the seeds of cloud and non-cloud
                if self.show_figure:
                    utils.show_seed_mask(
                        cloud_ml,
                        cloud_ml,
                        self.cloud_model_classes,
                        "Seed: " + "&".join(self.base_machine_learning),
                    )

                # physical rules and Control to make the rules combined dynamically
                if (i == 1) or self.physical_rules_dynamic:
                    self.physical.set_options([True, False], [True, False], [True, False])
                else:  # only use the physical rules determined by the initilization when we do not need to adjust the rules anymore
                    self.physical.set_options([options[0]], [options[1]], [options[2]])

                # select the cloud probability layer by the physical rules
                (prob_ph, options, thrd) = self.physical.select_cloud_probability(
                    cloud_ml,
                    label_cloud = label_cloud,
                    label_noncloud = label_noncloud,
                    show_figure = self.show_figure
                )
                # mask cloud by the physical rules
                cloud_ph = np.zeros(self.image.shape, dtype="uint8")
                if label_noncloud != 0: # only when it is not zero, we update the mask of noncloud
                    cloud_ph[prob_ph < thrd]= label_noncloud
                # see overlap_cloud_probability, in which we counted this included to mask clouds
                # have to be PCP pixels
                cloud_ph[(prob_ph >= thrd) & self.physical.pcp] = label_cloud
                # exclude extremely cold clouds when thermal band is available
                if self.image.data.exist("tirs1"):
                    cloud_ph[self.physical.cold_cloud] = label_cloud
                # set the mask of non-observed pixels to filled
                cloud_ph[self.image.filled] = label_filled

                if self.show_figure:
                    # make title with the options at the end (TTT)
                    utils.show_cloud_probability(
                        self.physical.prob_cloud,
                        self.physical.image.filled,
                        f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})"
                    )
                    utils.show_cloud_mask(
                        cloud_ph, self.cloud_model_classes, "Physical rules"
                    )

                # when the outcome is physical at the end, we do not run the machine learning classification one more time.
                if (i == self.max_iteration) & (outcome == "physical"):
                    self.cloud.append(cloud_ph == label_cloud)
                    return

                # continue to classify and tune the machine learning model
                if C.MSG_FULL:
                    print(f">>> tunning machine learning model {i:02d}/{self.max_iteration:02d}")
                # tune the unet and return the cloud mask and cloud probability layer by the updated model
                if self.tune_machine_learning == "unet":
                    (cloud_ml_update, _) = self.unet_cloud.tune(cloud_ph)  # as seed layer
                elif (self.tune_machine_learning == "randomforest") | (self.tune_machine_learning == "lightgbm"):
                    # point to the cloud model which will be used as the tuner
                    if self.tune_machine_learning == "lightgbm":
                        _tunner = self.lightgbm_cloud
                    elif self.tune_machine_learning == "randomforest":
                        _tunner = self.rf_cloud
                    # get the seed layer
                    _update_model = True # control to update model or not
                    if self.tune_seed == "disagree":
                        _cloud_ph_seed = cloud_ph.copy()
                        # make the disagreement pixels as the seed
                        if self.tune_machine_learning == "lightgbm": _cloud_ph_seed[cloud_ph==cloud_lightgbm] = label_filled
                        if self.tune_machine_learning == "randomforest": _cloud_ph_seed[cloud_ph==cloud_rf] = label_filled
                        # when no data remained
                        if np.count_nonzero(_cloud_ph_seed < label_filled) == 0:
                           _update_model = False
                    else:
                        _cloud_ph_seed = cloud_ph

                    # update model
                    if _update_model:
                        # update the training data, and retrain the random forest model
                        # append the new samples to the training data
                        if _tunner.tune_append_rate > 0:
                            _tunner.sample.update(
                                self.image.data.get(_tunner.predictors),
                                _tunner.predictors,
                                _cloud_ph_seed,
                                label_cloud=label_cloud,
                                label_fill=label_filled,
                                number=int(
                                    self._tunner.tune_append_rate
                                    * self._tunner.sample.number
                                ),
                                method="append",
                            )
                        # update the training data by replacing the samples
                        if _tunner.tune_update_rate > 0:
                            _tunner.sample.update(
                                self.image.data.get(_tunner.predictors),
                                _tunner.predictors,
                                _cloud_ph_seed,
                                label_cloud=label_cloud,
                                label_fill=label_filled,
                                number=int(
                                    _tunner.tune_update_rate
                                    * _tunner.sample.number
                                ),
                                method="replace",
                            )
                        # retrain the random forest model only when we have the samples updated
                        if (_tunner.tune_append_rate > 0) | (_tunner.tune_update_rate > 0):
                            _tunner.train()
                        (cloud_ml_update, _,_) = _tunner.classify()
                    else: # when no data remained
                        cloud_ml_update = cloud_ml.copy()

                # exclude the filled pixels by the real extent masking
                cloud_ml_update[self.image.filled] = (
                    label_filled  # set the mask of non-observed pixels to filled
                )

                if self.show_figure:
                    utils.show_cloud_mask(
                        cloud_ml_update, self.cloud_model_classes, self.tune_machine_learning
                    )

                # disagreement between two layers and update the cloud mask
                disagree_ml = (
                    1
                    - np.count_nonzero(
                        (cloud_ml_update == cloud_ml) & self.image.obsmask
                    )
                    / self.image.obsnum
                )
                # update to the new cloud mask and cloud probability layer
                cloud_ml = cloud_ml_update.copy()

                # update the cloud mask by the machine learning model
                if outcome == "classified":
                    # make a buffer to connect the cloud pixels when the subsampling is used
                    # this was designed to speed up the classification using pixel-based classifiers, but after testing, it will harm the classification results, so we do not use it
                    if (self.tune_machine_learning == "randomforest") and (self.rf_cloud.subsampling_size > 1):
                        self.cloud.append(utils.dilate(cloud_ml == label_cloud, radius=self.rf_cloud.subsampling_size-1))
                    elif (self.tune_machine_learning == "lightgbm") and (self.lightgbm_cloud.subsampling_size > 1):
                        self.cloud.append(utils.dilate(cloud_ml == label_cloud, radius=self.lightgbm_cloud.subsampling_size-1))
                    else:
                        self.cloud.append(cloud_ml == label_cloud)
                else:
                    self.cloud.append(cloud_ph == label_cloud)

                # reach to the end iteration
                if (i == self.max_iteration):
                    if C.MSG_FULL:
                        print(
                            ">>> stop iterating at the end"
                        )
                    return

                # stop by the disagreement rate
                if disagree_ml < self.disagree_rate:
                    if C.MSG_FULL:
                        print(
                            f">>> stop iterating with disagreement = {disagree_ml:.2f} less than {self.disagree_rate}"
                        )
                    return

                # stop if the cloud and non cloud seed pixels are not enough to represent 
                count_cloud = np.count_nonzero(cloud_ml == label_cloud)
                count_noncloud = np.count_nonzero(cloud_ml == label_noncloud)
                if (count_cloud < self.physical.min_clear) | (count_noncloud < self.physical.min_clear):
                    if C.MSG_FULL:
                        print(
                            f">>> stop iterating with less representive seed pixels for cloud = {count_cloud} for noncloud = {count_noncloud}"
                        )
                    return


    def mask_cloud_unet(self, probability="cloud") -> None:
        """mask clouds by UNet

        Args:
            probability (str, optional): "cloud": cloud prob. "noncloud": noncloud prob. or "none": not to extract the prob layer. "default": highest score for the classified results. Defaults to "none".
        """
        # init cloud layer
        self.cloud = BitLayer(self.image.shape)

        self.unet_cloud.load_model()
        _cloud, prob_ml = self.unet_cloud.classify(probability=probability)
        self.cloud.append(
            _cloud == self.unet_cloud.classes.index("cloud")
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(cloud_mask, self.cloud_model_classes, "UNet")
            utils.show_cloud_probability(
                prob_ml, self.unet_cloud.image.filled, "Cloud Probability"
            )

    def mask_cloud_lightgbm(self, probability="cloud") -> None:
        """
        Masks clouds in the image using LightGBM.
        This method initializes the LightGBM model if it is not already activated,
        creates a cloud mask layer, classifies the cloud probability, and updates
        the cloud mask. Optionally, it can display the cloud mask and cloud probability
        figures.
        Args:
            probability (str, optional): The probability type to use for cloud classification.
                                         Defaults to "cloud".
        Returns:
            None
        """
        
        # init model
        if not self.lightgbm_cloud.activated:
            self.lightgbm_cloud.load_model()

        # init cloud layer
        self.cloud = BitLayer(self.image.shape)
        
        # classify the cloud, and get the cloud probability layer only when we want to show the figure
        if self.show_figure:
            (_cloud, prob_ml, _) = self.lightgbm_cloud.classify(
                probability=probability
            )  # the cloud probability layer, its definition is based on the classes
        else: # just cloud layer returned
            (_cloud, _, _) = self.lightgbm_cloud.classify(
                probability=probability
            )
        # append to the final cloud layer
        self.cloud.append(
            _cloud == self.lightgbm_cloud.classes.index("cloud")
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud

        # check if we need to show the figure
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(cloud_mask, self.cloud_model_classes, "LightGBM")
            utils.show_cloud_probability(
                prob_ml, self.unet_cloud.image.filled, "Cloud Probability"
            )
        
    def mask_cloud_random_forest(self, probability="cloud") -> None:
        """Mask cloud by the random forest model

        Args:
            probability (str, optional): "cloud": cloud prob. "noncloud": noncloud prob. or "none": not to extract the prob layer. "default": highest score for the classified results. Defaults to "none".
        """
        # init cloud layer
        self.cloud = BitLayer(self.image.shape)

        (_cloud, prob_ml, _) = self.rf_cloud.classify(
            probability=probability
        )  # the cloud probability layer, its definition is based on the classes
        self.cloud.append(
            _cloud == self.rf_cloud.classes.index("cloud")
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud

        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(cloud_mask, self.cloud_model_classes, "Random Forest")
            utils.show_cloud_probability(
                prob_ml, self.unet_cloud.image.filled, "Cloud Probability"
            )

    def display_predictor(self, band, title=None, percentiles=None) -> None:
        _band = self.image.data.get(band)
        _band[self.image.filled] = np.nan
        # _band = np.interp(_band, np.nanpercentile(_band, percentiles), [0, 1])
        vrange = np.nanpercentile(_band, percentiles)
        utils.show_predictor(_band, self.image.filled, title, vrange= vrange)

    def display_image(
        self, bands=None, title=None, percentiles=None, path=None
    ) -> None:
        """Display a color image composed of specified bands.

        Args:
            bands (list, optional): List of band names to compose the color image. Defaults to ["red", "green", "blue"].
            title (str, optional): Title of the displayed image. Defaults to None.
            percentiles (list, optional): List of percentiles to use for contrast stretching. Defaults to [2, 98].
        """
        if bands is None:
            bands = ["red", "green", "blue"]
        if title is None:
            title = f"Color image ({bands[0]}, {bands[1]}, {bands[2]})"
        if percentiles is None:
            percentiles = [2, 98]
        rgb = utils.composite_rgb(
            self.image.data.get(bands[0]),
            self.image.data.get(bands[1]),
            self.image.data.get(bands[2]),
            self.image.obsmask,
            percentiles=percentiles,
        )
        utils.show_image(rgb, title, path)

    def save_mask(self, endname=None) -> None:
        """Save the mask to the specified path.

        Args:
            path (str): The path to save the mask.
            mask (ndarray): The mask to save.
            classes (list): The classes of the mask.
            title (str): The title of the mask.
            format (str, optional): The format of the mask. Defaults to "GTiff".
        """
        # get the mask
        emask = self.ensemble_mask
        # update the profile
        profile = self.image.profile.copy()
        profile["dtype"] = type(emask)  # update the dtype accordingly
        if endname is None:
            endname = self.algorithm
        # create the directory if it does not exist
        Path(self.image.destination).mkdir(parents=True, exist_ok=True)
        utils.save_raster(
            emask,
            profile,
            os.path.join(
                self.image.destination, self.image.name + "_" + endname.upper() + ".tif"
            ),
        )

    def save_model_metadata(self, path, running_time=0.0) -> None:
        """save model's metadata to a CSV file

        Args:
            path (_type_): _description_
            running_time (float, optional): The running time of the algorithm. Defaults to 0.0.
        """
        df_accuracy = pd.DataFrame.from_dict(
            [
                {
                    "image": self.image.name,
                    "model": self.algorithm,
                    "cloud_percentage": self.cloud_percentage,
                    "spectral_variation": self.physical.options[0],
                    "temperature_hot": self.physical.options[1],
                    "cirrus": self.physical.options[2],
                    "threshold": self.physical.threshold,
                    "running_time": running_time,
                }
            ],
            orient="columns",
        )
        # create the directory if it does not exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        print(df_accuracy)
        df_accuracy.to_csv(path)
        

    def save_accuracy(self, dataset, path, running_time=0.0, shadow=False):
        """Saves the accuracy metrics of the cloud and shadow masks to a CSV file.

        Parameters:
            dataset (str): The dataset name.
            endname (str, optional): The suffix to append to the output file name. Defaults to None.
            running_time (float, optional): The running time of the algorithm. Defaults to 0.0.
            shadow (bool, optional): Flag indicating whether to include shadow accuracy metrics. Defaults to False.

        Returns:
            None
        """
        # Function code goes here
        emask = self.ensemble_mask
        # read the manual mask
        mmask = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=emask.shape
        )
        # when we do not get the accruacy of shadow layer
        if not shadow:
            mmask[mmask == C.LABEL_SHADOW] = (
                C.LABEL_CLEAR
            )  # we consider shadow as clear
            emask[emask == C.LABEL_SHADOW] = (
                C.LABEL_CLEAR
            )  # we consider shadow as clear
        emask[emask == C.LABEL_LAND] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        emask[emask == C.LABEL_WATER] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        emask[emask == C.LABEL_SNOW] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        # same extent between the manual mask and the ensemble mask
        mmask[emask == C.LABEL_FILL] = C.LABEL_FILL  # same extent with the manual mask
        emask[mmask == C.LABEL_FILL] = C.LABEL_FILL  # same extent with the manual mask

        mmask = mmask[mmask != C.LABEL_FILL]
        emask = emask[emask != C.LABEL_FILL]

        # Cloud, Shadow, and Clear
        csc_overall = accuracy_score(mmask, emask)
        cloud_precision, shadow_precision = precision_score(
            mmask,
            emask,
            labels=[C.LABEL_CLOUD, C.LABEL_SHADOW],
            average=None,
            zero_division=1.0,
        )
        cloud_recall, shadow_recall = recall_score(
            mmask,
            emask,
            labels=[C.LABEL_CLOUD, C.LABEL_SHADOW],
            average=None,
            zero_division=1.0,
        )

        # Cloud, and Non-cloud (cloud shadow and clear)
        cn_overall = accuracy_score(mmask == C.LABEL_CLOUD, emask == C.LABEL_CLOUD)

        # Cloud percentage
        cloud_percentage_pred = np.count_nonzero(emask == C.LABEL_CLOUD) / len(emask)
        cloud_percentage_true = np.count_nonzero(mmask == C.LABEL_CLOUD) / len(mmask)

        # Cloud shadow percentage
        shadow_percentage_pred = np.count_nonzero(emask == C.LABEL_SHADOW) / len(emask)
        shadow_percentage_true = np.count_nonzero(mmask == C.LABEL_SHADOW) / len(mmask)

        # Number of observaiont pixels
        num_obs_pred = len(emask)
        num_obs_true = len(mmask)

        df_accuracy = pd.DataFrame.from_dict(
            [
                {
                    "image": self.image.name,
                    "cloud_percentage_pred": cloud_percentage_pred,
                    "cloud_percentage_true": cloud_percentage_true,
                    "shadow_percentage_pred": shadow_percentage_pred,
                    "shadow_percentage_true": shadow_percentage_true,
                    "cn_overall": cn_overall,
                    "csc_overall": csc_overall,
                    "cloud_precision": cloud_precision,
                    "cloud_recall": cloud_recall,
                    "shadow_precision": shadow_precision,
                    "shadow_recall": shadow_recall,
                    "num_obs_pred": num_obs_pred,
                    "num_obs_true": num_obs_true,
                    "running_time": running_time,
                }
            ],
            orient="columns",
        )
        # create the directory if it does not exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        print(df_accuracy)
        df_accuracy.to_csv(path)

    def mask_shadow_unet(self, probability="none"):
        """mask shadow by pretrained unet

        Args:
            probability (str, optional): can be "shadow", "noshadow", "none". Defaults to "none".

        Returns:
            nd.array(bool): shadow mask by pretrained unet
        """
        if not self.unet_shadow.activated:
            self.unet_shadow.load_model()
        shadow_ml, _ = self.unet_shadow.classify(probability=probability)
        shadow_ml = shadow_ml == self.unet_shadow.classes.index(
            "shadow"
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud
        return shadow_ml

    def mask_shadow_flood(self):
        """mask cloud shadows based on the flood method

        Returns:
            nd.array(bool): shadow mask by fill flooded
        """
        return flood_fill_shadow(
            self.image.data.get("nir"),
            self.image.data.get("swir1"),
            self.physical.abs_clear_land,
            self.image.obsmask,
        )

    def mask_shadow_geometry(self, potential="flood", clean_image = False):
        """
        Masks the shadow in the image using the specified potential algorithms.

        Args:
            potential (str or list, optional): Potential shadow detection algorithm(s) to use.
                If None, both "unet" and "flood" algorithms will be used. Defaults to None.
            false_cloud_layer (int, optional): The false layer value. Defaults to 0.
        Returns:
            None

        Raises:
            None
        """
        if (potential is None) or (potential.lower() == "both"):
            potential = ["UNet", "Flood"]
        else:
            potential = [potential]
        # potential shadow mask
        pshadow = np.zeros(self.image.obsmask.shape, dtype="uint8")
        for ialg in potential:
            if ialg.lower() == "flood":
                shadow_mask_binary = self.mask_shadow_flood()
            elif ialg.lower() == "unet":
                shadow_mask_binary = self.mask_shadow_unet()
            # add the shadow mask to the potential shadow mask
            pshadow = pshadow + shadow_mask_binary
            if self.show_figure:
                shadow_mask = shadow_mask_binary.copy().astype("uint8")
                shadow_mask[self.image.filled] = (
                    2  # use all list indicating the classes
                )
                utils.show_shadow_mask(
                    shadow_mask, ["nonshadow", "shadow", "filled"], ialg
                )
            del shadow_mask_binary
        
        # clean image for empty the memory
        if clean_image:
            self.image = None
        
        # normalize the potential shadow mask into [0, 1], as weitghted sum to compute the similarity between cloud and shadow
        pshadow = pshadow / len(potential)
        self.shadow = self.physical.match_cloud2shadow(
            self.cloud_object,
            self.cloud_region,
            pshadow
        )

        # clean image for empty the memory
        if clean_image:
            self.cloud_object = None
            self.cloud_region = None

    def display_fmask(self, path=None):
        """display the fmask, with clear, cloud, shadow, and fill"""
        utils.show_fmask(self.ensemble_mask, "Fmask", path)

    # %% major port of masking clouds
    def mask_cloud(self, algorithm=None):
        """Masks clouds in the image using the specified algorithm.

        Parameters:
            algorithm (str): The algorithm to use for cloud masking.
                Valid options are "physical", "randomforest", "unet", and "interaction".
                Defaults to "physical".

        Returns:
            it will update the cloud mask in the object
        """
        # if the algorithm is not provided, use the default algorithm
        if algorithm is None:
            algorithm = self.algorithm

        # mask cloud by the specified algorithm
        if algorithm == "physical":
            # mask cloud by the physical rules with default settings
            self.mask_cloud_physical()

        elif algorithm == "randomforest":
            self.mask_cloud_random_forest()
        
        elif algorithm =="lightgbm":
            self.mask_cloud_lightgbm()

        elif algorithm == "unet":
            self.mask_cloud_unet()

        elif algorithm == "interaction":
            # mask cloud by the interacted physical rules and machine learning model
            self.mask_cloud_interaction()

    def __init__(self, image_path: str = "", algorithm: str = "interaction", base: str = "unet", tune: str = "lightgbm"):
        """
        Initialize the Fmask object.

        Parameters:
        - image_path (str): The path to the image file.
        - algorithm (str): The algorithm to be used for cloud masking. Default is "interaction".

        Returns:
        - None
        """
        # set the package directory, which is the parent directory of the current file, as the root, to access the base pre-trained models
        self.dir_package = Path(__file__).parent.parent

        # initlize image object, that contains base information on this image
        if Path(image_path).stem.startswith("L"):
            self.image = Landsat(image_path)
        elif Path(image_path).stem.startswith("S"):
            self.image = Sentinel2(image_path)
            
        # which algorithm will be used for cloud masking
        self.algorithm = algorithm
        self.base_machine_learning = base
        self.tune_machine_learning = tune
        # init modules that will be used in the cloud masking
        self.init_modules()

        # optimize the configuration according to the spacecraft
        # self.optimize()


# %%
