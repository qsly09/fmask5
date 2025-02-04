"""The module to train the random forest model for cloud dection"""

# pylint: disable=line-too-long
import sys
from utils import collect_sample_pixel
import numpy as np
import glob
import os
import pickle
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import constant as C


class Dataset:
    """The class to load the training data for the random forest"""

    # %% Attributes
    directory = None  # the path to load trainning samples
    filename = "training_data.csv"  # the name of the training data file
    datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD"]  # datasets that will be used
    classes = [
        "noncloud",
        "cloud",
    ]  # the classes that will be used, respectively with values, 0 and 1
    # the strategy of selecting noncloud samples, can be "random" or "stratified"
    sampling_methods = [
        "stratified",
        "stratified",
    ]  # the sampling methods that will be used, respectively for the classes --- non-cloud and cloud
    exclude = (
        None  # the image that will be excluded from the dataset (only for testing)
    )
    data = None  # the training data in the form of pandas dataframe
    number = 10000  # total number of the training samples remained

    @property
    def length(self):
        """
        Returns the length of the data.

        If the data is None, returns 0.

        Returns:
            int: The length of the data.
        """
        if self.data is None:
            return 0
        else:
            return len(self.data)
    @property
    def full_predictors(self):
        """
        Get the predictors from the data.

        Returns:
            list: The predictors.
        """
        _full_predictors = self.data.columns
        # exclude the label column
        return _full_predictors.drop("label")

    def get(self, columns=None):
        """
        Retrieve data from the object.

        Parameters:
            columns (list or None): Optional list of column names to retrieve. If None, all columns (excluded label) are retrieved.

        Returns:
            numpy.ndarray: The retrieved data as a NumPy array.
        """
        if columns is None:
            return self.data[self.full_predictors].values
        else:
            return self.data[columns].values
    
    # %% Methods
    def set_datasets(self, datasets) -> None:
        """Set the datasets that will be used to load the training data.

        Args:
            datasets (list or str): The list of datasets in uppercase that will be used. If a string is provided, it will be converted to a list by splitting on commas and removing spaces.

        """
        # Convert string to list when a string is provided
        if isinstance(datasets, str):
            datasets = datasets.replace(" ", "").split(",")
        self.datasets = [dataset.upper() for dataset in datasets]

    def set_directory(self, directory) -> None:
        """Set the directory where the training data is stored.

        Args:
            directory (str): The directory where the training data is stored.

        """
        self.directory = directory

    def set_exclude(self, exclude) -> None:
        """Set the image that will be excluded from the dataset.

        Args:
            exclude (str): The image that will be excluded from the dataset.

        """
        self.exclude = exclude

    def set_number(self, number) -> None:
        """Set the number of samples to be used.

        Args:
            number (int): The number of samples to be used.

        """
        self.number = number

    def set_classes(self, classes) -> None:
        """Set the classes to be used.

        Args:
            classes (list): The classes to be used.

        """
        self.classes = classes

    def set_sampling_methods(self, sampling_methods) -> None:
        """
        Set the sampling methods for the object.

        Parameters:
        sampling_methods (list): A list of sampling methods.

        Returns:
        None
        """
        self.sampling_methods = sampling_methods

    def load(self, sequence=True) -> None:
        """Load dataset from the csv files within the directory"""
        # load packaged training dataset
        filepath_training_data = os.path.join(self.directory, "pixelbase.csv")
        if os.path.exists(filepath_training_data):
            if C.MSG_FULL:
                print(">>> loading all training data from database")
            training_data = pd.read_csv(os.path.join(filepath_training_data))
            # select the data based on the datasets
            if len(self.datasets) > 0:
                training_data = training_data[
                    training_data["dataset"].isin(self.datasets)
                ]
            # exclude the data based on the exclude
            if self.exclude is not None:
                if isinstance(self.exclude, str):
                    exclude = [self.exclude]
                else:
                    exclude = self.exclude
                for exc in exclude:
                    training_data = training_data[
                        ~training_data["image"].str.contains(exc)
                    ]
            # drop of the dataset and image columns that we do not need anymore
            training_data = training_data.drop(columns=["dataset", "image"])
        else:
            print(f">>> the pixel training data does not exist at {self.directory}")
            sys.exit(0)


        # change the column nnir as to nir, if it exists
        if "nnir" in training_data.columns:
            training_data.rename(columns={"nnir": "nir"}, inplace=True)
        # convert to index-based labels, starting from 0

        if sequence:
            for cls in self.classes:
                label_cls = self.classes.index(cls)
                if cls == "noncloud":
                    training_data["label"] = training_data["label"].replace(
                        C.LABEL_CLEAR, label_cls
                    )
                    training_data["label"] = training_data["label"].replace(
                        C.LABEL_SHADOW, label_cls
                    )
                    continue
                if cls == "cloud":
                    training_data["label"] = training_data["label"].replace(
                        C.LABEL_CLOUD, label_cls
                    )
                    continue
        self.data = training_data

    def select(self):
        """select the training data based on the cloud and non-cloud sampling strategy.
        The sampling_methods have to be same as to the order of classes
        """
        if C.MSG_FULL:
            print(">>> selecting training samples")

        # determine the number of samples for each class by equal, transfering to list or array
        numbers = [round(self.number / len(self.classes)) for _ in self.classes]
        # create an empty dataframe to store the training data
        training_data = pd.DataFrame()
        # select the samples based on the sampling strategy
        for i, cls in enumerate(self.classes):
            # index as the pixel value, starting from 0
            label_cls = self.classes.index(
                cls
            )
            num_cls = numbers[i]  # the number of the samples for the class
            # processing cloud
            if cls == "cloud":
                # select cloud pixels
                if self.sampling_methods[i].lower() == "random":
                    training_data = pd.concat(
                        [
                            training_data,
                            self.data[self.data["label"] == label_cls].sample(
                                n=num_cls, replace=False, random_state=C.RANDOM_SEED
                            ),
                        ]
                    )
                elif self.sampling_methods[i].lower() == "stratified":
                    # clear (cloud coverage < 35%)
                    num_cls_sub = round(num_cls / 3)
                    training_data = pd.concat(
                        [
                            training_data,
                            self.data[
                                (self.data["label"] == label_cls)
                                & (self.data["cloud_scene_percent"] < 0.35)
                            ].sample(
                                n=num_cls_sub, replace=False, random_state=C.RANDOM_SEED
                            ),
                        ]
                    )
                    # mid-cloud (≥ 35% and ≤ 65%)
                    training_data = pd.concat(
                        [
                            training_data,
                            self.data[
                                (self.data["label"] == label_cls)
                                & (self.data["cloud_scene_percent"] >= 0.35)
                                & (self.data["cloud_scene_percent"] <= 0.65)
                            ].sample(
                                n=num_cls_sub, replace=False, random_state=C.RANDOM_SEED
                            ),
                        ]
                    )
                    # cloudy (> 65%) images
                    training_data = pd.concat(
                        [
                            training_data,
                            self.data[
                                (self.data["label"] == label_cls)
                                & (self.data["cloud_scene_percent"] > 0.65)
                            ].sample(
                                n=num_cls_sub, replace=False, random_state=C.RANDOM_SEED
                            ),
                        ]
                    )
                continue
            # processing noncloud
            if cls == "noncloud":
                # select noncloud pixels
                if self.sampling_methods[i].lower() == "random":
                    training_data = pd.concat(
                        [
                            training_data,
                            self.data[self.data["label"] == label_cls].sample(
                                n=num_cls, replace=False, random_state=C.RANDOM_SEED
                            ),
                        ]
                    )
                elif self.sampling_methods[i].lower() == "stratified":
                    # unique values of landcover
                    # Get the unique values of 'B' column
                    cover_types = [1, 2, 4, 5, 7, 8, 9, 11]
                    num_cover = round(num_cls / len(cover_types))
                    for cover in cover_types:
                        training_data = pd.concat(
                            [
                                training_data,
                                self.data[
                                    (self.data["label"] == label_cls)
                                    & (self.data["landcover"] == cover)
                                ].sample(
                                    n=num_cover,
                                    replace=False,
                                    random_state=C.RANDOM_SEED,
                                ),
                            ]
                        )
                continue
        # combine the cloud and non-cloud and return it as the training data
        self.data = training_data.reset_index(drop=True)

    def update(self, data, bands, reference, label_cloud, label_fill, number=0, method = "replace"):
        """
        Updates the data in the object with new samples.

        Args:
            data (pd.DataFrame): The original data.
            reference (str): The reference value.
            label_cloud (int): The label for cloud pixels.
            label_fill (int): The label for fill pixels.
            number (int, optional): The number of samples to update. Defaults to 0.
            method (str, optional): The method to update the data. Defaults to "replace". or "append"

        Returns:
            None
        """
        if number > 0:
            nsamp = collect_sample_pixel(
                data,
                bands,
                reference,
                label_cloud=label_cloud,
                label_fill=label_fill,
                number=number,
                cloud_area=False,
                scene_prct=False,
            )
            # update the data
            # randomly update the number of the original sample set
            if method == "replace":
                if len(self.data) - len(nsamp) > 0:
                    self.data = pd.concat(
                        [
                            self.data.sample(
                                n=len(self.data) - len(nsamp), random_state=C.RANDOM_SEED
                            ),
                            nsamp,
                        ]
                    )
                else: # this happens when all samples are extracted based on local reference
                    self.data = nsamp
            elif method == "append":
                if len(nsamp) > 0:
                    self.data = pd.concat(
                        [
                            self.data,
                            nsamp,
                        ]
                    )

    def save(self, path):
        """save the data to the path defined

        Args:
            path (str): the path of .csv file
        """
        self.data.to_csv(path, index=False)

    def clear(self):
        """clear the data"""
        self.data = None

    def __init__(
        self,
        directory=None,
        datasets=None,
        classes=None,
        sampling_methods: list=None,
        filename="training_data.csv",
        number=5000,
        exclude=None,
    ):
        """
        Initializes the RFLib class.

        Args:
            datasets (type): The datasets to be used.
            classes (type): The classes to be classified.
            directory (type): The directory where the data is stored.
            filename (str, optional): The name of the training data file. Defaults to "training_data.csv".
            number (int, optional): The number of samples to be used. Defaults to 5000.
            exclude (str, optional): The image to be excluded. Defaults to None.
        """
        self.directory = directory
        self.set_datasets(datasets)
        self.classes = classes
        self.filename = filename
        self.number = number
        self.sampling_methods = sampling_methods
        self.exclude = exclude


class RandomForest(object):
    """Class of Random Forest for cloud detection

    Returns:
        Object: Random Forest model
    """
    
    image = None  # image object
    path = None  # path of base random forest model
    
    sample: Dataset = None
    
    subsampling_size = 1 # the size of the subsampling, 1: no subsampling, every 1 pixel is used
    subsampling_min = 0  # the mininum number of pixels to be classified at once for triggering physcial rules; 0 means no subsampling is used

    @property
    def activated(self):
        """Check if the object is activated"""
        return self.model is not None

    def set_database(self, database):
        """
        Set the classes for the object.

        Parameters:
        classes (list): A list of classes to be set.

        Returns:
        None
        """
        self.sample = database
  
    # %% Methods

    def train(self):
        """Train the model using the provided training data.

        Args:
            n_estimators (int, optional): The number of trees in the random forest. Defaults to 100.
        """

        if (
            self.predictors is None
        ):  # if we do not select the predictors, we will use all the columns except the label
            if C.MSG_FULL:
                print(f">>> training random forest {self.ntrees} tree based on {self.sample.length} samples")
                print(f">>> using {len(self.sample.data.head())} predictors: {self.sample.data.head()}")
            model = RandomForestClassifier(n_estimators=self.ntrees, random_state = C.RANDOM_SEED)
            model.fit(self.sample.get(), self.sample.get("label"))
        else:
            if C.MSG_FULL:
                print(f">>> training random forest {self.ntrees} tree based on {self.sample.length} samples")
                print(f">>> using {len(self.predictors)} predictors: {self.predictors}")
            model = RandomForestClassifier(n_estimators=self.ntrees, random_state = C.RANDOM_SEED)
            model.fit(self.sample.get(self.predictors), self.sample.get("label"))
        self.model = model # setup the pretrained model
        self.nsamples = self.sample.length # also update the number of samples accordingly

    def load_model(self):
        """ Load model from the provided path"""
        self.sample = pickle.load(open(self.path.replace(".pk", "_sample.pk"), "rb"))
        self.model = pickle.load(open(self.path, "rb"))

    def save(self, path = None):
        """Save the model to the provided path.

        Args:
            path (str): The path to save the model. e.g., rf_l7.pk
        """
        # save model
        if path is None:
            path = self.path
        pickle.dump(self.sample, open(path.replace(".pk", "_sample.pk"), "wb"))
        pickle.dump(self.model, open(path, "wb"))
        
            
    
    def save_importance(self, path = None):
        """Save the random forest model to the specified path.

        Args:
            path (str): The file path of the .csv. Defaults to None.
        """

        importances = (
            self.model.feature_importances_
        )  # to get importance for each variable
        # save the importance with the column name as to csv
        # create a data frame by using PREDICTIORS as columns and importances as values
        df_f_imp_xgb = pd.DataFrame(
            data=[importances], columns=self.predictors
        )
        # save the data frame as to csv
        if path is None:
            path = os.path.join(self.image.destination,  f"{self.image.name}_importance.csv")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df_f_imp_xgb.to_csv(path, index=False)

    
    def classify(self, probability="none", subsampling_mask = None, base = False) -> tuple:
        """
        Classify the image by the model and return the class and probability

        Args:
            probability (str, optional): "cloud": cloud prob. "noncloud": noncloud prob. or "none": not to extract the prob layer. "default": highest score for the classified results. Defaults to "none".

            base (bool, optional): If True, the base model is used. Defaults to False.
        Returns:
            tuple: A tuple containing the image class and probability.

        Raises:
            None

        Example usage:
            image_class, image_prob = classify(probability="cloud")
        """
        if C.MSG_FULL:
            print(">>> classifying the image by random forest model")
        # create the subsampling mask according to the subsampling size
        if subsampling_mask is None:
            if base:
                subsampling_mask = self.image.obsmask # do not subsample the image when the trigger is True
            else:
                subsampling_mask = np.zeros(self.image.shape, dtype=bool)
                subsampling_mask[::self.subsampling_size, ::self.subsampling_size] = True
                subsampling_mask = subsampling_mask & self.image.obsmask
        
        # get the pixels from the entire imagery
        sample_image_row_all, sample_image_col_all = np.where(subsampling_mask)

        # randomly select the pixels for classification at static random seed
        # only when the trigger is True and the subsampling_min is larger than 0
        if base and self.subsampling_min > 0:
            if len(sample_image_row_all) > self.subsampling_min:
                np.random.seed(C.RANDOM_SEED) # set the random seed for the subsampling
                idx = np.random.choice(len(sample_image_row_all), self.subsampling_min, replace=False)
                sample_image_row_all = sample_image_row_all[idx]
                sample_image_col_all = sample_image_col_all[idx]
                # delete the idx
                del idx
                # update the subsampling mask
                subsampling_mask = np.zeros(self.image.shape, dtype=bool)
                subsampling_mask[sample_image_row_all, sample_image_col_all] = True

        # init the masks
        image_class = np.zeros(
            self.image.shape, dtype=np.uint8
        )  # keep the same as to unet, 0: noncloud, 1: cloud
        if probability == "none":
            image_prob = None
        else:
            image_prob = np.zeros(self.image.shape, dtype=np.float32)

        # the labels of the model
        labs_model = self.model.classes_

        # classify the image by the model with subsets of the image
        for idx in range(0, len(sample_image_row_all), 100000):
            # subset of the image
            idx_start = idx
            idx_end = np.minimum(len(sample_image_row_all), idx + 100000)
            sample_image_row = sample_image_row_all[idx_start:idx_end]
            sample_image_col = sample_image_col_all[idx_start:idx_end]
            # get the inputting array
            # x_inputs = (self.image.data.get(self.predictors)[:,sample_image_row, sample_image_col]).T
            # x_inputs = np.swapaxes(x_inputs, 0, 1)
            pro_pred = self.model.predict_proba((self.image.data.get(self.predictors)[:,sample_image_row, sample_image_col]).T)
            label_pred = labs_model[np.argmax(pro_pred, axis=1)]

            # update fmask with the true label classified
            for lb in labs_model:
                image_class[
                    sample_image_row[label_pred == lb],
                    sample_image_col[label_pred == lb],
                ] = lb
            # del label_pred

            # update the image_prob
            if image_prob is not None:
                # get the highest score no matter what the class is
                if probability == "default":
                    pro_pred = np.max(pro_pred, axis=1)
                elif probability == "noncloud":
                    pro_pred = pro_pred[:, self.classes.index("noncloud")]
                elif probability == "cloud":
                    pro_pred = pro_pred[:, self.classes.index("cloud")]
                # update the image_prob
                image_prob[sample_image_row, sample_image_col] = pro_pred

        return image_class, image_prob, subsampling_mask

    def __init__(
        self,
        classes: list,
        predictors: list,
        ntrees: int = 100,
        max_depth: int = None,
        nsamples: int = 5000,
        tune_update_rate: float = 0.1,
        tune_append_rate: float = 0.0,
        subsampling_size = 1,
        subsampling_min = 0,
        path=None,
    ):
        """
        Initialize the RFLib object.

        Args:
            sampling_classes (dict[str], optional): List of class names with the sampling approach. Defaults to {"noncloud": "sample", "cloud": "sample"}.
            predictors (list, optional): List of predictors. Defaults to None.
            path (str, optional): Path to the model. Defaults to None.
        """

        self.classes = classes
        self.predictors = predictors
        self.nsamples = nsamples
        self.ntrees = ntrees # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.max_depth = max_depth # see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        self.path = path
        self.sample: Dataset = None
        self.model = None
        self.tune_update_rate = tune_update_rate
        self.tune_append_rate = tune_append_rate
        self.subsampling_size = subsampling_size
        self.subsampling_min = subsampling_min
