"""The module to train the random forest model for cloud dection"""

# pylint: disable=line-too-long
import numpy as np
import os
import pickle
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import constant as C
from rflib import Dataset # this is the dataset class for the random forest model


class LightGBM(object):
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

        # self.ntrees = ntrees # see n_estimators = 100 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        # self.num_leaves = num_leaves # num_leaves = 31 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        # self.min_data_in_leaf = min_data_in_leaf # min_data_in_leaf = 20 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        # self.max_depth = max_depth # max_depth = -1 (no limit) https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        model = lgb.LGBMClassifier(num_leaves = self.num_leaves, 
                                   max_depth = self.max_depth, 
                                   min_data_in_leaf = self.min_data_in_leaf, 
                                   n_estimators=self.ntrees, 
                                   random_state = C.RANDOM_SEED,
                                   n_jobs  = 1, # only use 1 core to process, since we can use parallel processing for each individual image
                                   verbose = -1) # no verbose, do not show the warnings in the progress
    
        if (
            self.predictors is None
        ):  # if we do not select the predictors, we will use all the columns except the label
            
            #
            if C.MSG_FULL:
                print(f">>> training lightgbm {self.ntrees} tree {self.num_leaves}, num_leaves, and {self.min_data_in_leaf} min_data_in_leaf based on {self.sample.length} samples")
                print(f">>> using {len(self.sample.data.head())} predictors: {self.sample.data.head()}")
            model.fit(self.sample.get(), self.sample.get("label"))
        else:
            if C.MSG_FULL:
                print(f">>> training lightgbm {self.ntrees} tree based on {self.sample.length} samples")
                print(f">>> using {len(self.predictors)} predictors: {self.predictors}")
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

        Returns:
            tuple: A tuple containing the image class and probability.

        Raises:
            None

        Example usage:
            image_class, image_prob = classify(probability="cloud")
        """
        if C.MSG_FULL:
            print(">>> classifying the image by lightgbm model")
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
        for idx in range(0, len(sample_image_row_all), 1000000):
            # subset of the image
            idx_start = idx
            idx_end = np.minimum(len(sample_image_row_all), idx + 1000000)
            sample_image_row = sample_image_row_all[idx_start:idx_end]
            sample_image_col = sample_image_col_all[idx_start:idx_end]
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
        num_leaves: int = 31,
        min_data_in_leaf: int = 20,
        max_depth: int = -1,
        nsamples: int = 10000,
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
        self.ntrees = ntrees # see n_estimators = 100 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        self.num_leaves = num_leaves # num_leaves = 31 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        self.min_data_in_leaf = min_data_in_leaf # min_data_in_leaf = 20 https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        self.max_depth = max_depth # max_depth = -1 (no limit) https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
        self.path = path
        self.sample: Dataset = None
        self.model = None
        self.tune_update_rate = tune_update_rate
        self.tune_append_rate = tune_append_rate
        self.subsampling_size = subsampling_size
        self.subsampling_min = subsampling_min
