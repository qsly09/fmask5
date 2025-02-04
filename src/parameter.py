"""manage the parameters of the Fmask algorithm"""

l8_predictor_full = [
    "coastal",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "tirs2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for full model

l8_predictor_cloud_phy = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "dem",
    "swo",
]

l8_predictor_cloud_pixel = [
    "coastal",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "tirs2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for cloud pixel model

l8_predictor_cloud_cnn = [
    "coastal",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "tirs2",
    "cirrus",
    "dem",
    "swo",
]  # predictors for cloud CNN model

l8_predictor_shadow_cnn = [
    "nir",
    "swir1",
    "dem"
]  # predictors for shadow CNN model

s2_predictor_full = [
    "coastal",
    "blue",
    "green",
    "red",
    "vre1",
    "vre2",
    "vre3",
    "wnir",
    "nir",
    "wv",
    "swir1",
    "swir2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "cdi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for full model

s2_predictor_cloud_phy = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "dem",
    "swo",
]  # predictors for full model

s2_predictor_cloud_pixel = [
    "coastal",
    "blue",
    "green",
    "red",
    "vre1",
    "vre2",
    "vre3",
    "wnir",
    "nir",
    "wv",
    "swir1",
    "swir2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "cdi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for cloud pixel model

s2_predictor_cloud_cnn = [
    "coastal",
    "blue",
    "green",
    "red",
    "vre1",
    "vre2",
    "vre3",
    "wnir",
    "nir",
    "wv",
    "swir1",
    "swir2",
    "cirrus",
    "dem",
    "swo",
]  # predictors for cloud CNN model

s2_predictor_shadow_cnn = [
    "nir",
    "swir1",
    "dem",
]  # predictors for shadow CNN model

# End of predictors

class Parameter:
    """A class representing a set of parameters.
    """

    def __init__(self, spacecraft, **kwargs):
        """
        Initialize the parameters with given keyword arguments.
        
        :param kwargs: Dictionary of parameters.
        """
        
        if spacecraft in ["LANDSAT_8", "LANDSAT_9"]:
            self.dataset = ["L8BIOME", "L8SPARCS", "L895CLOUD"]  # training datasets
            self.predictor_full = l8_predictor_full
            self.predictor_cloud_phy = l8_predictor_cloud_phy
            self.predictor_cloud_pixel = l8_predictor_cloud_pixel
            self.predictor_cloud_cnn = l8_predictor_cloud_cnn
            self.predictor_shadow_cnn = l8_predictor_shadow_cnn
            self.seed_level = 0
            self.epoch_tune = 5
            self.update_rate = 0.2
            if self.trigger == "unet":
                self.max_iteration = 1
                self.disagree_rate = 1.0
            elif self.trigger == "randomforest":
                self.max_iteration = 2
                self.disagree_rate = 0.05

            self.woc = 0.3  # Weight Of Cirrus
            self.threshold = 0.175  # threshold of masking cloud
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = 90  # the radius of erosion, unit: pixels
            return
        if spacecraft in ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7"]:
            self.dataset = l8_dataset
            self.predictor_full = l8_predictor_full
            self.predictor_cloud_phy = l8_predictor_cloud_phy
            self.predictor_cloud_pixel = l8_predictor_cloud_pixel
            self.predictor_cloud_cnn = l8_predictor_cloud_cnn
            self.predictor_shadow_cnn = l8_predictor_shadow_cnn
            self.seed_level = 0
            self.epoch_tune = 5
            self.update_rate = 0.2
            if self.tuner == "unet":
                self.max_iteration = 1
                self.disagree_rate = 1.0
            elif self.tuner == "randomforest":
                self.max_iteration = 2
                self.disagree_rate = 0.05

            self.woc = 0.0  # Weight Of Cirrus
            self.threshold = 0.1  # threshold of masking cloud
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = 150  # the radius of erosion, unit: pixels
         if spacecraft in ["SENTINEL-2A", "SENTINEL-2B"]:
            self.dataset = s2_dataset
            self.predictor_full = s2_predictor_full
            self.predictor_cloud_phy = s2_predictor_cloud_phy
            self.predictor_cloud_pixel = s2_predictor_cloud_pixel
            self.predictor_cloud_cnn = s2_predictor_cloud_cnn
            self.predictor_shadow_cnn = s2_predictor_shadow_cnn
            if self.trigger == "unet":
                self.seed_level = 25
            elif self.trigger == "randomforest":
                self.seed_level = 75
            self.epoch_tune = 5
            self.update_rate = 0.2
            self.max_iteration = 2
            if self.tuner == "unet":
                self.disagree_rate = 0.1
            elif self.tuner == "randomforest":
                self.disagree_rate = 0.25

            self.woc = 0.5  # Weight Of Cirrus
            self.threshold = 0.2  # threshold of masking cloud
            self.resolution = 20  # spatial resolution that we processing the image
            self.erosion_radius = 90  # the radius of erosion, unit: pixels

        for key, value in kwargs.items():
            setattr(self, key, value)

    def get(self, key, default=None):
        """
        Get the value of a parameter.
        
        :param key: The parameter name.
        :param default: The default value if the parameter does not exist.
        :return: The value of the parameter or default if not found.
        """
        return getattr(self, key, default)

    def set(self, key, value):
        """
        Set the value of a parameter.
        
        :param key: The parameter name.
        :param value: The new value of the parameter.
        """
        setattr(self, key, value)

    def to_dict(self):
        """
        Convert the parameters to a dictionary.
        
        :return: A dictionary representation of the parameters.
        """
        return self.__dict__

    def __repr__(self):
        """
        Return a string representation of the parameters.
        
        :return: A string representing the parameters.
        """
        return f"Parameters({self.to_dict()})"


# # Example usage:
# params = Parameters(learning_rate=0.01, batch_size=32, epochs=100)
# print(params)  # Output: Parameters({'learning_rate': 0.01, 'batch_size': 32, 'epochs': 100})

# # Access a parameter
# learning_rate = params.get('learning_rate')
# print(f"Learning Rate: {learning_rate}")  # Output: Learning Rate: 0.01

# # Set a new parameter
# params.set('momentum', 0.9)
# print(params)  # Output: Parameters({'learning_rate': 0.01, 'batch_size': 32, 'epochs': 100, 'momentum': 0.9})

# # Convert to dictionary
# params_dict = params.to_dict()
# print(params_dict)  # Output: {'learning_rate': 0.01, 'batch_size': 32, 'epochs': 100, 'momentum': 0.9}
