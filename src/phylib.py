"Physical rules to detect clouds"

import numpy as np
import utils
import pandas
from scipy.ndimage.filters import uniform_filter
from sklearn.linear_model import LinearRegression
import constant as C
from skimage.measure import label, regionprops
from satellite import Data
import copy


def mask_pcp(data: Data, satu):
    """mask possible cloud pixels (PCPs)

    Args:
        data (Data): datacube
        satu (2d array): saturation band

    Returns:
        bool: possible cloud pixels
    """
    # Basic test
    pcp = np.logical_and(
        np.logical_and(data.get("ndvi") < 0.8, data.get("ndsi") < 0.8),
        data.get("swir2") > 0.03,
    )

    # Temperature
    if data.exist("tirs1"):
        pcp = np.logical_and(pcp, data.get("tirs1") < 27)  # in degree

    # Whiteness test
    pcp = np.logical_and(pcp, data.get("whiteness") < 0.7)

    # Haze test
    pcp = np.logical_and(pcp, np.logical_or(data.get("hot") > 0, satu))

    # Ratio 4/5 test
    pcp = np.logical_and(
        pcp,
        (data.get("nir") / (data.get("swir1") + C.EPS)) > 0.75,
    )

    return pcp


def mask_snow(data: Data):
    """It takes every snow pixels including snow pixel under thin clouds or icy clouds

    Args:
        data (Data): datacube

    Returns:
        bool: snow/ice pixels
    """
    snow = np.logical_and(
        np.logical_and(
            data.get("ndsi") > 0.15,
            data.get("nir") > 0.11,
        ),
        data.get("green") > 0.1,
    )
    if data.exist("tirs1"):
        snow = np.logical_and(snow, data.get("tirs1") < 10)  # in degree
    return snow


def mask_abs_snow(data: Data, green_satu, snow, radius=167):
    """Select absolute snow/ice pixels using spectral-contextual for polar regions where large area of snow/ice (see Qiu et al., 2019)"

    Args:
        data (Data): datacube
        green_satu (bool): Saturation of the green band
        snow (bool): spectral-based snow/ice mask
        radius (int, optional): Kernel size in pixels. Defaults to 167.

    Returns:
        bool: snow/ice pixels
    """

    # radius = 2*radius + 1 # as to the window size which is used directly
    # green_var = uniform_filter((data.get("green")*data.get("green")).astype(np.float32), radius, mode='reflect') - np.square(uniform_filter((data.get("green")).astype(np.float32), radius, mode='reflect')) # must convert to float32 or 64 to get the uniform_filter
    # green_var[green_var<0] = C.EPS # Equal to 0
    # absnow = np.logical_and(np.logical_and(np.sqrt(green_var)*np.sqrt(radius*radius/(radius*radius-1))*(1-data.get("ndsi")) < 0.0009, snow), ~green_satu) # np.sqrt(green_var)*(1-ndsi) < 9 is SCSI
    # Note np.sqrt(radius*radius/(radius*radius-1)) is to convert it as same as that of matlab function, stdflit, see https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

    # only get snow/ice pixels from all potential snow/ice pixels, and
    # do not select the saturated pixels at green band which may be cloud!
    # Local standard deviation of image's ref: https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    # and https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    radius = 2 * radius + 1  # as to the window size which is used directly
    green_var = uniform_filter(
        (data.get("green").astype(np.float32)) ** 2,
        radius,
        mode="reflect",
    ) - np.square(
        uniform_filter(
            data.get("green").astype(np.float32),
            radius,
            mode="reflect",
        )
    )  # must convert to float32 or 64 to get the uniform_filter .astype(np.float32)
    green_var[green_var < 0] = C.EPS  # Equal to 0
    absnow = np.logical_and(
        np.logical_and(
            np.sqrt(green_var)
            * np.sqrt(radius**2 / (radius**2 - 1))
            * (1 - data.get("ndsi"))
            < 0.0009,
            snow,
        ),
        ~green_satu,
    )  # np.sqrt(green_var)*(1-ndsi) < 9 is SCSI
    # Note np.sqrt(radius**2/(radius**2-1)) is to convert it as same as that of matlab function, stdflit, see https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    return absnow


def split_to_zones(min_dem, max_dem, step_dem):
    """Splite the range from min to max into multiple zones, with a step of size. This version is some different from the Matlab version

    Args:
        min_dem (float): Mininum DEM
        max_dem (float): Maximum DEM
        step_dem (float): Step

    Returns:
        list of zones: _description_
    """
    splits = np.arange(min_dem, max_dem, step_dem)
    # in case we do not split the dem
    if len(splits) == 0:
        return None
    # regular processing
    zones = []
    for i, start in enumerate(splits):  # every 100 meters
        if i == 0:
            zone = [float("-inf"), start + step_dem]  # first zone
        elif i == len(splits) - 1:
            zone = [start, float("inf")]  # last zone
        else:
            zone = [start, start + step_dem]  # other zones
        zones.append(zone)
    return zones


def normalize_cirrus(cirrus, clear, obsmask=None, dem=None, dem_min=None, dem_max=None):
    """Normlize cirrus band based on DEM

    Args:
        cirrus (float): Cirrus band in reflectance, that will be varied
        clear (bool): Clear pixels in the image
        obsmask (bool, optional): Observation mask. Defaults to None.
        dem (float, optional): DEM data. Defaults to None.
        dem_min (float, optional): Mininum DEM in the image. Defaults to None.
        dem_max (float, optional): Maxinum DEM in the image. Defaults to None.

    Returns:
        float: Normalized cirrus band
    """

    if np.any(
        np.logical_and(clear, obsmask)
    ):  # if no abs clear pixels, just return the original cirrus band
        _cirrus = cirrus.copy()  # do not alter the original data
        # clear = np.logical_and(clear, obsmask) # updated with the mask at the first
        if dem is not None:
            if dem_min < dem_max:  # all the pixels are not the same, in case zeros
                dem_zones = split_to_zones(dem_min, dem_max, 100)
                clear_base = 0  # clear pixels's based cirrus band reflectance
                for zone in dem_zones:
                    dem_zone = np.logical_and(
                        obsmask, np.logical_and(dem > zone[0], dem <= zone[1])
                    )
                    if dem_zone.any():
                        clear_zone = np.logical_and(clear, dem_zone)
                        if clear_zone.any():
                            clear_base = get_percentile(_cirrus, clear_zone, 2)
                            # 2 percentile as dark pixel # updated as the newest base
                        _cirrus[dem_zone] = (
                            _cirrus[dem_zone] - clear_base
                        )  # in the case when we do not get the base, we can fetch up the previous one as base to continue to
            else:
                _cirrus = _cirrus - get_percentile(
                    _cirrus, np.logical_and(clear, obsmask), 2
                )
        else:  # cirrus band is adjusted to 0-level even when there is no DEM
            _cirrus = _cirrus - get_percentile(
                _cirrus, np.logical_and(clear, obsmask), 2
            )
            # 2 percentile as dark pixel
        _cirrus[_cirrus <= 0] = (
            0  # the normalized cirrus value will be set as 0 when it is negative
        )
        return _cirrus
    else:
        return cirrus


def normalize_temperature(temperature, dem, dem_min, dem_max, clear_land, number):
    """Normalize the thermal band

    Args:
        temperature (float): Thermal band in degree
        dem (float): DEM in meter
        dem_min (float, optional): Mininum DEM in the image. Defaults to None.
        dem_max (float, optional): Maxinum DEM in the image. Defaults to None.
        clear_land (float): Clear land pixels
        number (int): Number of pixels for fitting model

    Returns:
        float: Normalized thermal
    """
    _temperature = temperature.copy()  # do not alter the original data
    # clear_land = np.logical_and(clear_land, obsmask) # updated with the mask at the first
    # only use the high confident levels to estimate
    [temp_low, temp_high] = np.percentile(
        _temperature[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    samples = stratify_samples(
        np.logical_and(
            clear_land,
            np.logical_and(_temperature >= temp_low, _temperature <= temp_high),
        ),  # do not wnat change its original value
        dem,
        dem_min,  # np.percentile(dem[obsmask], 0.001)
        dem_max,  # 0.001 for excluding the noises of DEM data
        number=number,  # total number of sample pixels we expected
        step=300,
        distance=15,
    )  # mininum distance: unit: pixels
    if (
        len(samples) > 20
    ):  # 10 by 2 (parameters, i.e., slope and intercept); A common rule of thumb is to have at least 10 times as many samples as there are parameters to be estimated.
        # Estmate the lapse_rate of temperature by Ordinary least squares Linear Regression.
        reg = LinearRegression().fit(
            dem[samples.row, samples.column].reshape(-1, 1),
            _temperature[samples.row, samples.column],
        )
        # reg.coef_[0] is the slope, which is treated as the rate, that must be negative pysically
        if (
            reg.coef_[0] < 0
        ):  # only when the rate is negative pysically, we normalize the temperature band
            _temperature = _temperature - reg.coef_[0] * (
                dem - dem_min
            )  # the pixels located at higher elevation will be normalized to be one with higher temperature
    return _temperature, temp_low, temp_high


def stratify_samples(clear, data, data_min, data_max, step, number, distance=15):
    """stratified sampling DEM

    Args:
        clear (bool): Clear pixels
        data (2d array): Data inputted
        data_min (float): Mininum data
        data_max (float): Maxinum data
        step (float): Step interval to split data
        number (int, optional): Number of pixels seleceted.
        distance (int, optional): Mininum distance among the sampling pixels in pixels. Defaults to 15.

    Returns:
        dataframe: sampling pixels in row and column
    """

    data_zones = split_to_zones(data_min, data_max, step)

    df_sample_selected = []
    if (
        data_zones is not None
    ):  # this was because sometimes min value == max value in DEM
        number = round(number / len(data_zones))
        # equal samples in each stratum

        # we create a basic layer for sampling, we locatd a pixel every min_distance, and later on we only picked up the sample overlapped this base layer. This will be faster than previous MATLAB version, but similar performance
        if distance > 0:
            sampling_dist_base_layer = np.zeros(
                data.shape, dtype=bool
            )  # create a boolean base layer
            sampling_dist_base_layer[
                np.ix_(
                    range(0, sampling_dist_base_layer.shape[0], distance),
                    range(0, sampling_dist_base_layer.shape[1], distance),
                )
            ] = True  # _ix can quickly construct index arrays that will index the cross product.

        # at each isolate zone, we go to fetch the samples
        for zone in data_zones:
            clear_zone = np.logical_and.reduce(
                [clear, zone[0] < data, data <= zone[1]]
            )  # at the isolated zone
            if clear_zone.any():  # if any pixels located
                if distance > 0:  # over the mininum distrance base layer
                    clear_zone = np.logical_and(clear_zone, sampling_dist_base_layer)
                df_clear_zone = pandas.DataFrame(
                    np.argwhere(
                        clear_zone
                    ),  # pick up the pixels located in the isolated zone
                    columns=["row", "column"],
                )  # creating df object with columns specified
                del clear_zone
                df_clear_zone = df_clear_zone.sample(
                    np.min(
                        [number, len(df_clear_zone.index)]
                    ),  # when the number of data is smaller than that we expected
                    random_state=C.RANDOM_SEED,  # static seed for random
                    ignore_index=True,
                )  # unnecessery to update the index

                # append to the final
                df_sample_selected.append(df_clear_zone)
                del df_clear_zone
        if len(df_sample_selected) > 0:
            df_sample_selected = pandas.concat(df_sample_selected)
    return df_sample_selected


def mask_water(data: Data, obsmask, snow):
    """the spectral-based water mask (works over thin cloud)

    Args:
        data (Data): datacube
        obsmask (bool): Observation mask of the image
        snow (float): snow/ice mask

    Returns:
        bool: water pixels
    """
    water = np.logical_or(
        np.logical_and(
            np.logical_and(
                data.get("ndvi") > 0,
                data.get("ndvi") < 0.1,
            ),
            data.get("nir") < 0.05,
        ),
        np.logical_and(
            data.get("ndvi") < 0.01,
            data.get("nir") < 0.11,
        ),
    )
    water = np.logical_and(water, obsmask)
    # the swo-based water mask
    if water.any() and data.exist(
        "swo"
    ):  # when water pixels were identifed and the swo is available
        swo = data.get("swo")  # to get the layer
        # low level (17.5%) to exclude the commssion errors as water.
        # 5% tolerances.
        swo_thrd = np.percentile(swo[water], C.LOW_LEVEL, method="midpoint") - 5
        # merge the spectral-based water mask and the swo-based water mask
        if swo_thrd > 0:
            water = np.logical_or(
                water, np.logical_and(swo > swo_thrd, ~snow)
            )  # get the swo-based water mask, exclude snow/ice over ocean
    return water


def probability_cirrus(cirrus):
    """Cloud probability of cirrus

    Args:
        cirrus (float): Cirrus band reflectance

    Returns:
        float: Cloud probability of cirrus
    """
    prob_cir = np.clip(cirrus / 0.04, 0, None)
    return prob_cir


def probability_land_varibility(data: Data, green_satu, red_satu):
    """Cloud probability of varibility for land

    Args:
        data (Data): datacube
        green_satu: bool: Saturation of the green band
        red_satu: bool: Saturation of the red band

    Returns:
        float: Cloud probability of varibility for land
    """
    # prob_land_var = 1 - np.amax(
    #     [np.abs(ndvi), np.abs(ndsi), np.abs(ndbi), whiteness], axis=0
    # )
    # fixed the saturated visible bands for NDVI and NDSI
    _ndvi = np.where(
        np.bitwise_and(green_satu, data.get("ndvi") < 0), 0, data.get("ndvi")
    )
    _ndsi = np.where(
        np.bitwise_and(red_satu, data.get("ndsi") < 0), 0, data.get("ndsi")
    )
    return 1 - np.maximum(
        np.maximum(
            np.maximum(
                np.abs(_ndvi),
                np.abs(_ndsi),
            ),
            np.abs(data.get("ndbi")),
        ),
        data.get("whiteness"),
    )


def probability_land_temperature(temperature, clear_land):
    """Calculate the probability of land temperature based on the given temperature array and clear land mask.

    Parameters:
    temperature (ndarray): Array of temperature values.
    clear_land (ndarray): Boolean mask indicating clear land areas.

    Returns:
    prob_land_temp (ndarray): Array of probabilities of land temperature.
    temp_low (float): Lower threshold temperature.
    temp_high (float): Upper threshold temperature.
    """

    [temp_low, temp_high] = np.percentile(
        temperature[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    temp_low = temp_low - 4  # 4 C-degrees
    temp_high = temp_high + 4  # 4 C-degrees
    prob_land_temp = (temp_high - temperature) / (temp_high - temp_low)
    prob_land_temp = np.clip(prob_land_temp, 0, None)
    return prob_land_temp


def probability_water_temperature(temperature, clear_water):
    """Calculate the probability of water temperature based on the given temperature array and clear water mask.

    Parameters:
    temperature (numpy.ndarray): Array of temperature values.
    clear_water (numpy.ndarray): Boolean mask indicating clear water pixels.

    Returns:
    numpy.ndarray: Array of probabilities of water temperature.

    """
    prob_water_temp = (
        np.percentile(temperature[clear_water], C.HIGH_LEVEL) - temperature
    ) / 4  # 4 degree to normalize
    prob_water_temp = np.clip(prob_water_temp, 0, None)
    return prob_water_temp


def probability_water_brightness(data: Data):
    """
    Calculate the probability of water brightness for each pixel in the input data.

    Parameters:
    data (Data): The input data containing the necessary bands.

    Returns:
    prob_water_bright (numpy.ndarray): The probability of water brightness for each pixel.
    """
    prob_water_bright = data.get("swir1") / 0.11
    prob_water_bright = np.clip(prob_water_bright, 0, 1)
    return prob_water_bright


def probability_land_brightness(data, clear_land):
    """
    Calculate the probability of land brightness for a given set of hot and clear land values.

    Parameters:
    data (Data): The input data containing the necessary bands.
    clear_land (numpy.ndarray): Array of clear land values.

    Returns:
    numpy.ndarray: Array of probabilities of land brightness.

    """
    [hot_low, hot_high] = np.percentile(
        data.get("hot")[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    hot_low = hot_low - 0.04  # 0.04 reflectance
    hot_high = hot_high + 0.04  # 0.04 reflectance
    prob_land_bright = (data.get("hot") - hot_low) / (hot_high - hot_low)
    prob_land_bright = np.clip(
        prob_land_bright, 0, 1
    )  # 1  # this cannot be higher 1 (maybe have commission errors from bright surfaces).
    return prob_land_bright


def flood_fill_shadow(nir_full, swir1_full, abs_land, obsmask, threshold=0.02):
    """
    Masks potential shadow areas in the input images based on flood fill method.

    Parameters:
        nir_full (numpy.ndarray): Array representing the NIR band image.
        swir1_full (numpy.ndarray): Array representing the SWIR 1 band image.
        abs_land (numpy.ndarray): Array representing the land mask.
        obsmask (numpy.ndarray): Array representing the observation mask.
        thershold (float, optional): The threshold value for the shadow mask. Defaults to 0.02.

    Returns:
        numpy.ndarray: Array representing the mask of potential shadow areas.
    """

    # mask potential shadow using flood fill method in NIR and SWIR 1 band
    nir_surface_background = np.percentile(nir_full[abs_land], C.LOW_LEVEL)
    swir1_surface_background = np.percentile(swir1_full[abs_land], C.LOW_LEVEL)
    return (
        np.minimum(
            utils.imfill(nir_full, obsmask, fill_value=nir_surface_background)
            - nir_full,
            utils.imfill(swir1_full, obsmask, fill_value=swir1_surface_background)
            - swir1_full,
        )
        > threshold
    )


def get_percentile(data, obsmask, pct):
    """get percentile value

    Args:
        data (number): data layer
        obsmask (bool): observation mask
        pct (number): percentile value

    Returns:
        number: _description_
    """
    return np.percentile(data[obsmask], pct)


def compute_cloud_probability_layers(image, min_clear):
    """Compute cloud probability layers according to the datacube

    Args:
        image (Object): Landsat or Sentinel-2 object
        saturation (2d array): Saturation mask
        min_clear (number, optional): Mininum number for further analyse.

    Returns:
        various varibles: all components regarding cloud probabilities
    """

    # shared variables (between functions) with start of the dash _
    _dem_min, _dem_max = get_percentile(
        image.data.get("dem"), image.obsmask, [0.001, 99.999]
    )

    # _dem_max = get_percentile(data[bands.index("dem"), :, :], image.obsmask, 99.999)

    ## identify Potential Cloud Pixels (PCPs)
    pcp = mask_pcp(image.data, image.get_saturation(band="visible"))
    snow = mask_snow(image.data)

    # Exclude absolute snow pixels out of PCPs
    if np.count_nonzero(snow) >= np.square(
        10000 / image.resolution
    ):  # when the snow pixels are enough
        # exclude absolute snow/ice pixels
        pcp[
            mask_abs_snow(
                image.data,
                image.get_saturation(band="green"),
                snow,
                radius=np.ceil(5000 / image.resolution),
            )
        ] = False

    # Appen thin cloud pixels to PCPs
    if image.data.exist("cirrus"):
        cirrus = normalize_cirrus(
            image.data.get("cirrus"),
            ~pcp,
            obsmask=image.obsmask,
            dem=image.data.get("dem"),
            dem_min=_dem_min,
            dem_max=_dem_max,
        )
        pcp = np.logical_or(
            pcp, cirrus > 0.01
        )  # Update the PCPs with cirrus band TOA > 0.01, which may be cloudy as well

    # ABS CLEAR PIXELs with the observation extent
    # This can save the 'not' operation at next processings
    abs_clr = np.logical_and(
        ~pcp, image.obsmask
    )  # convert pcp as clear and updated it with the obs. mask

    # Seperate absolute clear mask into land and waster groups
    # mask water no mater if we go further to analyze the prob.
    water = mask_water(image.data, image.obsmask, snow=snow)

    # that will be used if the thermal band is available
    surface_low_temp, surface_high_temp = None, None

    # Start to anaylze cloud prob.
    if np.count_nonzero(abs_clr) <= min_clear:
        # Case 1: special case when there are lots of cloudy pixels
        # mask_cloud  = ~mask_pcp # all PCPs were masked as cloud directly because of no enought clear pixels for further analyses
        # mask_water = utils.init_mask(IMAGE.obsmask, dtype = 'bool', defaultvalue = 0) # no water pixels, in order to merge all the layers into 1 layer at the end
        # mask potential shadow
        activated = False
        # pcp, done above
        lprob_var = None
        lprob_temp = None
        wprob_temp = None
        wprob_bright = None
        prob_cirrus = None
        # water, done above
        # snow, done above
    else:
        # Case 2: regular cases, with cloud prob.

        # Fist of all, to check out cirrus band thermal -based probs. and once they are conducted, we can empty the data
        # Cloud probability: thin cloud (or cirrus) probability for both water and land
        if image.data.exist("cirrus"):
            prob_cirrus = probability_cirrus(cirrus)
            del cirrus  # that can be deleted
        else:
            prob_cirrus = 0

        abs_land = np.logical_and(~water, abs_clr)
        # Check the number of absolute clear pixels, and when not enough, fmask goes back to pick up all
        if np.count_nonzero(abs_land) <= min_clear:
            abs_land = abs_clr
        abs_water = np.logical_and(water, abs_clr)
        del abs_clr

        # OVER LAND #
        # Cloud probability: varibility probability over land
        lprob_var = probability_land_varibility(
            image.data,
            image.get_saturation(band="green"),
            image.get_saturation(band="red"),
        )

        # Cloud probability: temperature and brightness probability over land
        if image.data.exist("tirs1"):
            tirs1, surface_low_temp, surface_high_temp = normalize_temperature(
                image.data.get("tirs1"),
                image.data.get("dem"),
                _dem_min,
                _dem_max,
                abs_land,
                min_clear,
            )
            lprob_temp = probability_land_temperature(
                tirs1, abs_land
            )  # that has been normalized
        else:
            lprob_temp = probability_land_brightness(image.data, abs_land)

        # END of LAND #

        # OVER WATER #
        if np.any(
            abs_water
        ):  # Only when the clear water pixels are identified, this will be triggered.
            # Cloud probability: temperature probability over water
            if image.data.exist("tirs1"):
                wprob_temp = probability_water_temperature(tirs1, abs_water)
                del tirs1
            else:
                wprob_temp = 1.0
            # Cloud probability: brightness probability over water
            wprob_bright = probability_water_brightness(image.data)
        else:
            wprob_temp = None
            wprob_bright = None
        # END of WATER #

        activated = True
    return (
        activated,
        pcp,
        lprob_var,
        lprob_temp,
        wprob_temp,
        wprob_bright,
        prob_cirrus,
        water,
        snow,
        surface_low_temp,
        surface_high_temp,
    )


def combine_cloud_probability(
    var, tmp, cir, prob_var, prob_temp, prob_cirrus, woc, mask_absclear, adjusted=True
):
    """combine cloud probability layer according to the physical rules

    Args:
        var (bool): True to control the variation probability
        tmp (bool): True to control the temporal probability
        cir (bool): True to control the cirrus probability
        prob_var (2d array): variation probability
        prob_temp (2d array): temporal probability
        prob_cirrus (2d array): cirrus probability
        woc (number): Weight of cirrus probability
        mask_absclear (2d array in bool): Clear pixels
        adjust (bool, optional): Adjust the cloud probability. Defaults to True.

    Returns:
        2d array: Combined cloud probability
    """
    # combine cloud probabilities for land or water
    if var and tmp:
        prob = prob_var * prob_temp
    elif var:
        # copy the prob_var to prob, and then update the prob according to the mask_absclear
        prob = prob_var
    elif tmp:
        prob = prob_temp
    else:
        prob = 0
    if cir:
        prob = prob + woc * prob_cirrus
    if adjusted:
        prob = prob - clear_probability(prob, mask_absclear)
        prob[prob < 0] = 0 # we will set it as 0 in final prob.
    # prob[mask_absclear] = 0 # no need. exclude abs clear pixels from the cloud probability as 0
    return prob


def convert2seedgroups(
    mask_prob, seed, label_cloud, label_noncloud, bin_width=0.025, equal_num=False
):
    """
    Convert the mask probabilities of seed pixels into two groups: cloud and non-cloud.

    Args:
        mask_prob (numpy.ndarray): Array of mask probabilities.
        seed (numpy.ndarray): Array of seed labels.
        label_cloud: Label for cloud pixels.
        label_noncloud: Label for non-cloud pixels.
        bin_width (float, optional): Width of the probability bins. Defaults to 0.025.
        equal_num (bool, optional): Whether to have an equal number of seed pixels between cloud and non-cloud.
                                   Defaults to True.

    Returns:
        tuple: A tuple containing:
            - seed_cloud_prob (numpy.ndarray): Array of mask probabilities for cloud seed pixels.
            - seed_noncloud_prob (numpy.ndarray): Array of mask probabilities for non-cloud seed pixels.
            - prob_range (list): Range of mask probabilities.

    """
    seed_cloud_prob = mask_prob[seed == label_cloud].flatten()
    seed_noncloud_prob = mask_prob[seed == label_noncloud].flatten()

    # 0.05% was used to exclude potential anormly data
    # merge the two groups of seed pixels
    prob_range = np.percentile(
        np.concatenate([seed_cloud_prob, seed_noncloud_prob]),
        q=[0.05, 99.95],
    )
    prob_range = [
        np.floor(prob_range[0] / bin_width) * bin_width,
        np.ceil(prob_range[1] / bin_width) * bin_width,
    ]  # make the range into 0.025 level

    # in case when the prob_min is very close to prob_max, the bins will be empty
    prob_range[0] = min(
        prob_range[0], 0
    )  # to make sure we have the full range of probability
    prob_range[1] = max(
        prob_range[1], 1
    )  # to make sure we have the full range of probability

    # adjust the values to avoid the discarded pixels in final total number
    seed_cloud_prob[seed_cloud_prob < prob_range[0]] = prob_range[0]
    seed_cloud_prob[seed_cloud_prob > prob_range[1]] = prob_range[1]

    if equal_num:
        np.random.seed(C.RANDOM_SEED)
        # same number of seed pixels between cloud and non-cloud
        if seed_cloud_prob.size > seed_noncloud_prob.size:
            seed_cloud_prob = seed_cloud_prob[
                np.random.choice(
                    seed_cloud_prob.size, seed_noncloud_prob.size, replace=False
                )
            ]
        elif seed_cloud_prob.size < seed_noncloud_prob.size:
            seed_noncloud_prob = seed_noncloud_prob[
                np.random.choice(
                    seed_noncloud_prob.size, seed_cloud_prob.size, replace=False
                )
            ]

    return seed_cloud_prob, seed_noncloud_prob, prob_range


def overlap_cloud_probability(
    seed_cloud_prob,
    seed_noncloud_prob,
    prob_range=None,
    prob_bin=0.025,
    threshold=0,
    split=True,
):
    """find the overlapping density of cloud and non-cloud pixels and the optimal thershold to separate them

    Args:
        mask_prob (2d array): physical-based cloud probability
        mask_seed (2d array): mask of cloud and non-cloud layer, in which the cloud is 1 and the non-cloud is 0, and filled pixels are provided with a different value
        label_cloud (int, optional): pixel value indicating cloud. Defaults to 1.
        label_noncloud (int, optional): pixel value indicating noncloud. Defaults to 0.
        prob_range (list, optional): range of cloud probability. Defaults to [0, 1].
        prob_bin (float, optional): width of bins of cloud probability. Defaults to 0.025.
        threshold (float, optional): threshold to separate cloud and non-cloud pixels. Defaults to 0.05.
        split (bool, optional): split cloud and non-cloud pixels by a thershold. Defaults to True.

    Returns:
        number, number: overlapping density and optimal thershold
    """

    # calculate the density hist for each dataset with specified matching bin edges
    if prob_range is None:
        prob_range = [0, 1]
    [prob_min, prob_max] = prob_range

    if prob_max > prob_min:
        bins_thrd = np.arange(
            prob_min, prob_max + prob_bin, prob_bin
        )  # + prob_bin to make sure the prob_max is included
    else:  # in case when the prob_min is same as to prob_max, the bins will be empty
        bins_thrd = [prob_min, prob_min + prob_bin]

    bins_cloud, _ = np.histogram(seed_cloud_prob, bins=bins_thrd)
    bins_noncloud, _ = np.histogram(seed_noncloud_prob, bins=bins_thrd)
    # e.g., if bans are [0, 1, 2, 3], the counts will be [0, 1) [1, 2) [2, 3)
    # total_pixels = len(seed_cloud_prob) + len(seed_noncloud_prob)

    # calculate the overlapping density
    # bins_overlap = np.min([bins_cloud, bins_noncloud], axis=0)
    over_rate = (np.min([bins_cloud, bins_noncloud], axis=0)).sum() / (
        len(seed_cloud_prob) + len(seed_noncloud_prob)
    )

    if split:  # to get the optimal thershold
        # search optimal thershold to seperate cloud and non-cloud pixels
        # as we do not want to miss cloud pixels in the physical layer, we will set the threshold_buffer as 0.95
        thrd_record = bins_thrd[
            -1
        ]  # default with maximum error rate by setting the thershold as 1.0
        num_errors_record = (
            bins_cloud.sum()
        )  # in this case, all cloudy pixels were misclassified as non-cloud
        bins_thrd = bins_thrd[:-1]  # remove the last bin's right boundary
        for i, thrd in enumerate(bins_thrd):
            # max_overlap = np.max([bins_overlap[0:i].sum(), bins_overlap[i:].sum()])
            # count # of cloud pixels on left of the x-axis and # of non-cloud pixels on right of the x-axis, and if their sum moves to be smaller, the result is optimal
            # the thershold of segmenting clouds is "> thrd" rather than ">= thrd", because the thershold is the left boundary of the bin (included), but the right boundary is not included.
            # cloud seed pixels were counted by "< thrd" and non-cloud seed pixels were counted by ">= thrd"
            num_errors = (
                bins_cloud[
                    0:i
                ].sum()  # cloud pixels on the left of the thershold, not included the boundary
                + bins_noncloud[
                    i:
                ].sum()  # non-cloud pixels on the right of the thershold, included the boundary
            )

            # the optimal thershold is the one that can make the overlapping density as low as possible
            # i.e., the thershold is altered only when the error rate reduced 5% compared to the previous one
            if (
                num_errors_record - num_errors
            ) > threshold * num_errors_record:  # same as to (num_errors - num_errors_record)/num_errors_record < -threshold
                num_errors_record = num_errors.copy()
                thrd_record = thrd.copy()
        return over_rate, thrd_record
    return over_rate, None


def clear_probability(prob, clear):
    """Calculate the clear probability based on the given probability array and clear mask.

    Parameters:
    prob (numpy.ndarray): The probability array.
    clear (numpy.ndarray): The clear mask.

    Returns:
    float: The clear probability.

    """
    return np.percentile(prob[clear], C.HIGH_LEVEL)


# define functions inside
def shift_by_sensor(coords, height, view_zenith, view_azimuth, resolution):
    """
    Shifts the given coordinates based on the sensor parameters.

    Args:
        coords (numpy.ndarray): Array of coordinates to be shifted.
        height (float): Height of the sensor.
        view_zenith (float): Zenith angle of the sensor's view.
        view_azimuth (float): Azimuth angle of the sensor's view.
        resolution (float): Resolution of the sensor.

    Returns:
        numpy.ndarray: Array of shifted coordinates.
    """
    shift_dist = (
        height * np.tan(view_zenith) / resolution
    )  # in shifting pixels over the plate
    coords[:, 1] = coords[:, 1] + shift_dist * np.cos(
        np.pi / 2 - view_azimuth
    )  # x-axis horizontal column
    coords[:, 0] = coords[:, 0] + shift_dist * -np.sin(
        np.pi / 2 - view_azimuth
    )  # y_axis vertical  row
    return coords


def shift_by_solar(coords, height, solar_elevation, solar_azimuth, resolution):
    """
    Shifts the given coordinates based on solar elevation and azimuth.

    Parameters:
    - coords (numpy.ndarray): Array of coordinates to be shifted.
    - height (float): Height of the object.
    - solar_elevation (float): Solar elevation angle in radians.
    - solar_azimuth (float): Solar azimuth angle in radians.
    - resolution (float): Resolution of the image.

    Returns:
    - numpy.ndarray: Array of shifted coordinates.
    """
    shift_dist = (height / np.tan(solar_elevation)) / resolution  # in pixels
    coords[:, 1] = coords[:, 1] - shift_dist * np.cos(
        solar_azimuth - np.pi / 2
    )  # x-axis horizontal column
    coords[:, 0] = coords[:, 0] - shift_dist * np.sin(
        solar_azimuth - np.pi / 2
    )  # y_axis vertical  row

    # if solar_azimuth < np.pi:
    #     coords[:, 1] = coords[:, 1] - shift_dist * np.cos(
    #         solar_azimuth - np.pi / 2
    #     )  # x-axis horizontal column
    #     coords[:, 0] = coords[:, 0] - shift_dist * np.sin(
    #         solar_azimuth - np.pi / 2
    #     )  # y_axis vertical  row
    # else:
    #     coords[:, 1] = coords[:, 1] + shift_dist * np.cos(
    #         solar_azimuth - np.pi / 2
    #     )  # x-axis horizontal column
    #     coords[:, 0] = coords[:, 0] + shift_dist * np.sin(
    #         solar_azimuth - np.pi / 2
    #     )  # y_axis vertical  row
    return coords


def project_dem2plane(ele, solar_elevation, solar_azimuth, resolution, mask_filled):
    """
    Projects a digital elevation model (DEM) to a plane based on solar elevation and azimuth.

    Args:
        ele (numpy.ndarray): The digital elevation model.
        solar_elevation (float): The solar elevation angle in degrees.
        solar_azimuth (float): The solar azimuth angle in degrees.
        resolution (float): The resolution of the DEM in meters.
        mask_filled (numpy.ndarray): A mask indicating filled pixels in the DEM.

    Returns:
        tuple: A tuple containing three arrays:
            - PLANE2IMAGE_ROW (numpy.ndarray): A matrix storing the mapping between the plane and the image (row indices).
            - PLANE2IMAGE_COL (numpy.ndarray): A matrix storing the mapping between the plane and the image (column indices).
            - PLANE_OFFSET (numpy.ndarray): The offset applied to the plane coordinates to make them positive.
    """

    ele = ele - np.percentile(
        ele[~mask_filled], 0.1
    )  # relative elevation 0.1 is to avoid the outlier
    # get the coordinates of all the dem pixels
    image_coords = np.argwhere(np.ones_like(ele, dtype=bool))
    plane_coords = shift_by_solar(
        image_coords.copy(),
        ele[image_coords[:, 0], image_coords[:, 1]],
        solar_elevation,
        solar_azimuth,
        resolution,
    )

    # create new array to preserve the plane_coords as positive
    PLANE_OFFSET = np.min(plane_coords, axis=0)
    plane_coords = plane_coords - PLANE_OFFSET  # make the plane_coords as positive
    PLANE_SHAPE = np.max(plane_coords, axis=0) + 1

    # create a matrix to store the mapping between the plane and the image
    PLANE2IMAGE_ROW = np.zeros(PLANE_SHAPE, dtype=np.int32)
    PLANE2IMAGE_COL = np.zeros(PLANE_SHAPE, dtype=np.int32)
    # convert to integer by round
    image_coords = np.round(image_coords).astype(np.int32)
    # append the mapping between the plane and the image
    PLANE2IMAGE_ROW[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 0]
    PLANE2IMAGE_COL[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 1]
    return PLANE2IMAGE_ROW, PLANE2IMAGE_COL, PLANE_OFFSET


def segment_cloud_objects(cloud, min_area=3, buffer2connect=0, exclude=None, exclude_method = 'any'):
    """
    Segment cloud objects in the given cloud image.

    Parameters:
    - cloud: numpy.ndarray
        The cloud image to segment.
    - min_area: int, optional
        The minimum area (in pixels) for a cloud object to be considered.
    - exclude: numpy.ndarray
        The exclude base layer 
    - exclude_method: str, optional
        The method to exclude the cloud objects. Defaults to 'any'. or 'all'
        if any cloud pixels overlap with this exclude layer, the cloud will be excluded from the cloud_regions.
        if all cloud pixels overlap with this exclude layer, the cloud will be excluded from the cloud_regions.

    Returns:
    - cloud_objects: numpy.ndarray
        The labeled cloud objects.
    - cloud_regions: list of skimage.measure._regionprops.RegionProperties
        The region properties of the segmented cloud objects.
    """
    if (
        buffer2connect > 0
    ):  # connect neighbor clouds to match cloud shadows at same time
        cloud_dilated = utils.dilate(cloud, radius=buffer2connect)
        cloud_objects = label(
            cloud_dilated, background=0, return_num=False, connectivity=None
        )
        cloud_objects[~cloud] = 0
    else:
        cloud_objects = label(cloud, background=0, return_num=False, connectivity=None)

    cloud_regions = regionprops(cloud_objects)
    if min_area > 0:
        cloud_regions = [
            icloud for icloud in cloud_regions if icloud.area >= min_area
        ]  # filter out the very small clouds

    if exclude is not None:
        if exclude_method == 'any':
            cloud_regions = [
                icloud
                for icloud in cloud_regions
                if not any(exclude[icloud.coords[:, 0], icloud.coords[:, 1]])
            ]
        elif exclude_method == 'all':
            cloud_regions = [
                icloud
                for icloud in cloud_regions
                if not all(exclude[icloud.coords[:, 0], icloud.coords[:, 1]])
            ]
        # revise cloud_objects as well
        cloud_objects = np.zeros_like(cloud_objects) # initialize the cloud_objects as 0
        for icloud in cloud_regions:
            cloud_objects[icloud.coords[:, 0], icloud.coords[:, 1]] = icloud.label

    return cloud_objects, cloud_regions


def match_cloud2shadow(
    cloud_regions,
    cloud_objects,
    pshadow,
    mask_filled,
    view_zenith,  # in degree
    view_azimuth,  # in degree
    solar_elevation,  # in degree
    solar_azimuth,  # in degree
    resolution,
    similarity=0.10,
    sampling_cloud=100000, # number of sampling pixels to find the shadow, in order to speed up the process. the value 0 means to use all the pixels
    thermal=None,
    surface_temp_low=None,
    surface_temp_high=None,
    ele=None,
    PLANE2IMAGE_ROW=None,
    PLANE2IMAGE_COL=None,
    PLANE_OFFSET=None,
    apcloud=False,
):
    """
    Matches the cloud mask with the shadow mask to identify cloud shadows.

    Args:
        cloud_objects (ndarray): Binary cloud mask.
        pshadow (ndarray): shadow mask with weight.
        mask_filled (ndarray): Filled mask.
        view_zenith (ndarray): View zenith angles in degrees.
        view_azimuth (ndarray): View azimuth angles in degrees.
        solar_elevation (float): Solar elevation angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        resolution (float): Resolution of the image.
        thermal (ndarray, optional): Thermal band data. Defaults to None.
        surface_temp_low (float, optional): Lower threshold for surface temperature. Defaults to None.
        surface_temp_high (float, optional): Upper threshold for surface temperature. Defaults to None.
        ele (ndarray, optional): Digital Elevation Model (DEM) data. Defaults to None.
        apcloud (ndarray, optional): Approve clouds identified by the cloud mask. Defaults to None.

    Returns:
        ndarray: Binary mask indicating the matched shadow areas.
    """
    # make sure the random sampling is the same fore reproducibility
    np.random.seed(C.RANDOM_SEED)
    pcloud = cloud_objects > 0
    pshadow[pcloud] = 0  # exclude cloud from the potential shadow

    # pcloud[mask_filled] = 0  # exclude the filled pixels from the cloud mask for sure
    # check the thermal band is included
    THERMAL_INCLUDED = thermal is not None
    DEM_PROJECTION = ele is not None
    NEIGHBOR_HEIGHT = (
        True  # control the height of the cloud object by the neighbor cloud object
    )
    solar_elevation = np.deg2rad(solar_elevation)  # convert to radian
    solar_azimuth = np.deg2rad(solar_azimuth)  # convert to radian

    # max similarity between the cloud object and the cloud shadow object
    similarity_min = similarity  # thershold of approving shadows
    similarity_max = 0.95
    similarity_buffer = 0.95
    num_close_clouds = 14 # see MFask, Qiu et al., 2017, RSE 

    # number of inward pixes (90m) for cloud base temperature
    num_edge_pixels = int(90 / resolution)  # in pixels
    rate_elapse = 0.0065  # enviromental lapse rate 6.5 degrees/km in degrees/meter
    rate_dlapse = 0.0098  # dry adiabatic lapse rate 9.8 degrees/km   in degrees/meter
    rate_dlapse_reduced = 0.001  # a reduced wet adiabatic lapse rate of − 1 K km− 1

    # read angles
    # solar angle in radian, that we just use the scene-center angle, because at each pixel, the solar angle is not varied a lot
    # and, it will be time-comsuming if we read the solar angle for each pixel

    # image size
    image_height, image_width = pshadow.shape
    shadow_mask_matched = np.zeros(pshadow.shape, dtype="bool")
    cloud_mask_matched = np.zeros(pshadow.shape, dtype="bool")

    # height interval for finding the cloud shadow
    cloud_height_interval = 2  # move 2 pixel at a time
    cloud_height_interval = cloud_height_interval * (
        resolution * np.tan(solar_elevation)
    )  # in meters

    # DEM projection
    if DEM_PROJECTION:
        PLANE_SHAPE = PLANE2IMAGE_ROW.shape

    if NEIGHBOR_HEIGHT:
        # create a matrix to store the distance between the cloud objects
        record_cloud_centroids = np.array([cloud.centroid for cloud in cloud_regions])
        # cloud height
        record_cloud_base_heights = np.zeros(len(cloud_regions), dtype=np.float32)

    # search cloud shadow from the cloud object closer to the center of the image
    # sort the cloud objects by the distance to the center of the image
    # find centeroid of the mask_filled
    centeroid_filled = np.array(regionprops((~mask_filled).astype(int))[0].centroid)
    cloud_regions = sorted(
        cloud_regions,
        key=lambda x: np.sum(np.square(centeroid_filled - x.centroid)),
    )

    # iterate the cloud objects by enumate loop
    for icloud, cloud in enumerate(cloud_regions):
        # print('Cloud: ', icloud)
        # find the cloud height by the neighbor cloud object
        record_close_cloud_base_height = (
            0.0  # zeros value will not start to find the cloud height nearby
        )
        if (
            NEIGHBOR_HEIGHT & icloud >= num_close_clouds
        ):  # start to find the cloud height nearby
            # find the closest cloud object based on the distance between the cloud object and the cloud object
            close_cloud_heights = record_cloud_base_heights[
                np.argsort(
                    np.sum(
                        np.abs(record_cloud_centroids[0:icloud] - cloud.centroid),
                        axis=1,
                    )
                )
            ]
            # remove the zero height
            close_cloud_heights = close_cloud_heights[close_cloud_heights != 0]
            if (
                len(close_cloud_heights) >= num_close_clouds
            ):  # when the number of close cloud objects is enough again
                close_cloud_heights = close_cloud_heights[
                    0:num_close_clouds
                ]  # get the first num_close_clouds
                if (
                    np.std(close_cloud_heights) >= 1000
                ):  # when the heights are very different
                    record_close_cloud_base_height = 0.0  #
                else:
                    record_close_cloud_base_height = np.percentile(
                        close_cloud_heights, 85
                    )  # higher level of the cloud heights
                    if (
                        record_close_cloud_base_height
                        <= cloud_height_min | record_close_cloud_base_height
                        >= cloud_height_max
                    ):
                        record_close_cloud_base_height = 0.0

        # in meters. 200m to 12km for cloud height usually
        cloud_height_min, cloud_height_max = 200.00, 12000.00

        # assume object is round and cloud_radius is radius of the cloud object
        cloud_radius = np.sqrt(cloud.area / 2 * np.pi)  # in pixels

        # down-sampling the big cloud object
        if (sampling_cloud > 0) and (cloud.area > sampling_cloud):
            csampling = np.random.choice(
                cloud.coords.shape[0], sampling_cloud, replace=False
            )
        else:
            csampling = np.arange(cloud.coords.shape[0])

        cloud_coords = cloud.coords.copy()

        # narrow the cloud height range according to the thermal band if it is available
        if THERMAL_INCLUDED:
            # obtain the thermal of the cloud object according to the cloud coordinates
            cloud_temp = thermal[cloud_coords[:, 0], cloud_coords[:, 1]]

            if cloud_radius > num_edge_pixels:  # work for big cloud object only
                ## to get the percentage of the cloud object that will be used to obtain the base temperature of the cloud object
                # prct_cloud = 100*np.square(cloud_radius - num_erosion_pixels)/np.square(cloud_radius)
                cloud_temp_base = np.percentile(
                    cloud_temp,
                    100
                    * np.square(cloud_radius - num_edge_pixels)
                    / np.square(cloud_radius),
                )
                # put the edge of the cloud the same value as cloud_base_temp
                cloud_temp[cloud_temp > cloud_temp_base] = cloud_temp_base
            else:
                # prct_cloud = 0
                cloud_temp_base = np.min(cloud_temp)

            # do not adjust the cloud height range for warm cloud object
            if (
                cloud_temp_base <= surface_temp_low
            ):  # reverse the condition "cloud_temp_base > surface_temp_low"
                # narrow the cloud height range according to the thermal band
                cloud_height_min = np.maximum(
                    cloud_height_min,
                    (surface_temp_low - cloud_temp_base) / rate_dlapse,
                )  # here we already minused 4 degree ahead
                cloud_height_max = np.minimum(
                    cloud_height_max,
                    (surface_temp_high - cloud_temp_base) / rate_dlapse_reduced,
                )  # a reduced wet adiabatic lapse rate of − 1 K km− 1  here we already plused 4 degree ahead

        if DEM_PROJECTION:
            # get the surface elevation underneath the cloud object
            cloud_surface_ele = np.percentile(
                ele[cloud_coords[:, 0], cloud_coords[:, 1]], C.HIGH_LEVEL
            )

        # recording variable for similarity between the cloud object and the cloud shadow
        record_similiarity = 0.0
        record_cloud_base_height = 0.0
        record_num_matched = 0

        # view angles for cloud object
        cloud_view_zenith = np.deg2rad(
            view_zenith[cloud_coords[:, 0], cloud_coords[:, 1]]
        )  # in radian
        cloud_view_azimuth = np.deg2rad(
            view_azimuth[cloud_coords[:, 0], cloud_coords[:, 1]]
        )  # in radian

        # iterate the cloud height from cloud_height_min to cloud_height_max
        for cloud_base_height in np.arange(
            cloud_height_min, cloud_height_max, cloud_height_interval
        ):
            # when thermal available, create 3D cloud object with the cloud height according to the thermal band
            if THERMAL_INCLUDED:
                cloud_height = (
                    cloud_temp_base - cloud_temp[csampling]
                ) * rate_elapse + cloud_base_height
            else:
                cloud_height = cloud_base_height
            # make it as new variable to keep the original cloud object
            coords = cloud_coords[csampling]
            # calculate the cloud's coords
            coords = shift_by_sensor(
                coords,
                cloud_height,
                cloud_view_zenith[csampling],
                cloud_view_azimuth[csampling],
                resolution,
            )

            # when dem is available, adjust the base height of the cloud object relative to the reference plane
            if DEM_PROJECTION:
                coords = shift_by_solar(
                    coords,
                    cloud_height + cloud_surface_ele,
                    solar_elevation,
                    solar_azimuth,
                    resolution,
                )  # relative to the reference plane
                # convert the coordinates from the plane to the image based on the plane_coords
                # find the coords within the plane_coords
                coords = coords - PLANE_OFFSET  # make the plane_coords as positive
                # the id list of the pixels out of the image
                list_coords_outside = (
                    (coords[:, 0] < 0)
                    | (coords[:, 0] >= PLANE_SHAPE[0])
                    | (coords[:, 1] < 0)
                    | (coords[:, 1] >= PLANE_SHAPE[1])
                )
                coords = coords[
                    ~list_coords_outside
                ]  # remove the pixels out of the image (reference plane)
                # convert the coords from the plane to the image which keep same array structure as the coords
                coords = np.array(
                    [
                        PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1]],
                        PLANE2IMAGE_COL[coords[:, 0], coords[:, 1]],
                    ]
                ).T
            else:
                coords = shift_by_solar(
                    coords, cloud_height, solar_elevation, solar_azimuth, resolution
                )

            # the id list of the pixels out of the image
            list_coords_outside = (
                (coords[:, 0] < 0)
                | (coords[:, 0] >= image_height)
                | (coords[:, 1] < 0)
                | (coords[:, 1] >= image_width)
            )
            coords = coords[~list_coords_outside]  # remove the pixels out of the image

            num_out_image = np.count_nonzero(list_coords_outside)

            # count the number of false pixels
            # the pixels over the cloud or shadow layer (exclude original cloud)
            shadow_projected = (
                cloud_objects[coords[:, 0], coords[:, 1]] != cloud.label
            )  # that include other clouds and clear pixels
            num_match2shadow = np.sum(
                shadow_projected * pshadow[coords[:, 0], coords[:, 1]]
            )  # here shadow_mask_binary has been merged with potential cloud and potential shadow together prior to
            num_match2cloud = np.count_nonzero(
                shadow_projected & pcloud[coords[:, 0], coords[:, 1]]
            )  # here cloud_mask_binary has been merged with potential cloud and potential shadow together prior to
            num_match2filled = np.count_nonzero(
                mask_filled[coords[:, 0], coords[:, 1]]
            )  # here mask_filled has been merged with potential cloud and potential shadow together prior to

            # 0.5 is used to punlish the projected pixels over cloud or filled layer, where there is no way to determine the potential cloud shadow
            num_pixels_matched = (
                num_match2shadow
                + 0.5 * (num_match2cloud + num_match2filled)
                + num_out_image
            )

            # number that is the total pixel (exclude original cloud)
            num_pixels_total = np.count_nonzero(shadow_projected) + num_out_image

            # similarity
            similarity_matched = num_pixels_matched / (num_pixels_total + C.EPS)

            # if we have found the cloud shadow, and the neighbor cloud height is lower than the previous one, then stop the iteration
            if not (
                record_num_matched > 0
                and record_close_cloud_base_height > 0
                and record_close_cloud_base_height < cloud_base_height
            ):
                # update the similarity recorded in this iteration
                if (
                    similarity_matched >= record_similiarity * similarity_buffer
                    and similarity_matched < similarity_max
                ):
                    if similarity_matched > record_similiarity:
                        record_similiarity = similarity_matched
                        record_cloud_base_height = cloud_base_height
                    continue  # continue the iteration
                else:
                    if record_similiarity >= similarity_min:
                        # a shadow was found
                        record_num_matched = record_num_matched + 1

                        # allow to continue to reach the height of the neighbor cloud object if the similarity is higher than the recorded one
                        if cloud_base_height < record_close_cloud_base_height:
                            if (
                                similarity_matched >= record_similiarity
                                or similarity_matched >= similarity_max
                            ):
                                record_similiarity = similarity_matched
                                record_cloud_base_height = cloud_base_height
                            continue  # continue the iteration
                        # stop the iteration
                    else:
                        record_similiarity = 0.0  # reset the similarity if the value of similarity is too small
                        continue  # continue the iteration

            if record_num_matched == 0:
                break  # stop the iteration if no cloud shadow is found

            # if the code reaches here, it means we have found the cloud shadow finally
            # when thermal available, create 3D cloud object with the cloud height according to the thermal band
            if THERMAL_INCLUDED:
                cloud_height = (
                    cloud_temp_base - cloud_temp  # use all the pixels
                ) * rate_elapse + record_cloud_base_height
            else:
                cloud_height = record_cloud_base_height

            # calculate the cloud's coords
            coords = (
                cloud.coords
            )  # at the last time, we use this variable, so it is ok when it is altered without copy()
            # approved clouds
            if apcloud:
                cloud_mask_matched[coords[:, 0], coords[:, 1]] = True

            coords = shift_by_sensor(
                coords,
                cloud_height,
                cloud_view_zenith,
                cloud_view_azimuth,
                resolution,
            )

            # when dem is available, adjust the base height of the cloud object relative to the reference plane
            if DEM_PROJECTION:
                coords = shift_by_solar(
                    coords,
                    cloud_height + cloud_surface_ele,
                    solar_elevation,
                    solar_azimuth,
                    resolution,
                )  # relative to the reference plane
                # convert the coordinates from the plane to the image based on the plane_coords
                # find the coords within the plane_coords
                coords = coords - PLANE_OFFSET  # make the plane_coords as positive
                # the id list of the pixels out of the image
                list_coords_outside = (
                    (coords[:, 0] < 0)
                    | (coords[:, 0] >= PLANE_SHAPE[0])
                    | (coords[:, 1] < 0)
                    | (coords[:, 1] >= PLANE_SHAPE[1])
                )
                coords = coords[
                    ~list_coords_outside
                ]  # remove the pixels out of the image (reference plane)
                # convert the coords from the plane to the image which keep same array structure as the coords
                coords = np.array(
                    [
                        PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1]],
                        PLANE2IMAGE_COL[coords[:, 0], coords[:, 1]],
                    ]
                ).T
            else:
                coords = shift_by_solar(
                    coords, cloud_height, solar_elevation, solar_azimuth, resolution
                )
                # the id list of the pixels out of the image
                list_coords_outside = (
                    (coords[:, 0] < 0)
                    | (coords[:, 0] >= image_height)
                    | (coords[:, 1] < 0)
                    | (coords[:, 1] >= image_width)
                )
                coords = coords[
                    ~list_coords_outside
                ]  # remove the pixels out of the image

            # recording
            shadow_mask_matched[coords[:, 0], coords[:, 1]] = True
            if (
                cloud_radius > num_edge_pixels
            ):  # not for small cloud object, its height will be used to assign the cloud height
                record_cloud_base_heights[icloud] = cloud_base_height
            record_num_matched = record_num_matched + 1
            # stop the iteration
            break
    return shadow_mask_matched, cloud_mask_matched


class Physical:
    """Physical model for cloud detection"""

    image = None
    activated = None # indicate it was initiated or not, and then turn to False or True
    pcp = None
    lprob_var = None
    lprob_temp = None
    wprob_temp = None
    wprob_bright = None
    prob_cirrus = None
    water = None
    snow = None
    surface_temp_low = None  # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
    surface_temp_high = None

    # options to control the cloud probabilities
    # and by default, all options are True
    options = [True, True, True]
    options_var = [True, False]
    options_temp = [True, False]
    options_cirrus = [True, False]

    # the minimum number of clear pixels to start up the cloud probability model
    # as well as the minimum number for representing clear surface pixels, which is used to normalize the thermal band
    min_clear = 40000
    
    sampling_cloud = 40000  # also optimal after testing it based on L8BIOME dataset. number of sampling pixels to find the cloud, in order to speed up the process. the value 0 means to use all the pixels
    similarity = 0.30  # max similarity between the cloud object and the cloud shadow object

    @property
    def abs_clear(self):
        """clear pixels

        Returns:
            2d array in bool: True for clear pixels, including land and water
        """
        return np.logical_and(
            self.image.obsmask, ~self.pcp
        )  # relying one the image object, we can get the obsmask

    @property
    def abs_clear_land(self):
        """clear land pixels

        Returns:
            2d array in bool: True for clear land pixels
        """
        _abs_clear_land = np.logical_and(~self.water, self.abs_clear)
        if (
            np.count_nonzero(_abs_clear_land) < self.min_clear
        ):  # in case we do not have enought clear land pixels
            _abs_clear_land = self.abs_clear
        return _abs_clear_land

    @property
    def abs_clear_water(self):
        """clear water pixels

        Returns:
            2d array in bool: True for clear water pixels
        """
        return np.logical_and(self.water, self.abs_clear)

    @property
    def prob_variation(self):
        """
        Calculate the cloud probability variation for land and water

        Returns:
            float: The cloud probability variation.
            None: If the method is not activated.
        """
        if self.activated:
            # record the orginal options
            options_var_temp = self.options_var.copy()
            options_temp_temp = self.options_temp.copy()
            options_cirrus_temp = self.options_cirrus.copy()

            # generate the cloud probability for temperature
            self.options_var = [False]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_temp, _, _ = self.select_cloud_probability(adjusted=False)

            # generate the cloud probability for spectral variation
            self.options_var = [True]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_var, _, _ = self.select_cloud_probability(adjusted=False)
            prob_var = prob_var / (
                prob_temp + C.EPS
            )  # select_cloud_prob does not support the rule only var

            # restore the orginal options
            self.options_var = options_var_temp.copy()
            self.options_temp = options_temp_temp.copy()
            self.options_cirrus = options_cirrus_temp.copy()

            return prob_var
        return None

    @property
    def prob_temperature(self):
        """
        Calculate the cloud probability temperature for land and water

        Returns:
            float: The cloud probability temperature.
            None: If the method is not activated.
        """
        if self.activated:
            # record the orginal options
            options_var_temp = self.options_var.copy()
            options_temp_temp = self.options_temp.copy()
            options_cirrus_temp = self.options_cirrus.copy()

            # generate the cloud probability for temperature
            self.options_var = [False]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_temp, _, _ = self.select_cloud_probability(adjusted=False)

            # restore the orginal options
            self.options_var = options_var_temp.copy()
            self.options_temp = options_temp_temp.copy()
            self.options_cirrus = options_cirrus_temp.copy()

            return prob_temp
        else:
            return None

    @property
    def prob_cloud(self):
        """Calculate the cloud probability.

        Returns:
            float: The cloud probability layer with the recorded options.
            None: If the method is not activated.
        """
        if self.activated:
            cloud_prob, _, _ = self.select_cloud_probability(
                seed=None,
                options_var=[self.options[0]],
                options_temp=[self.options[1]],
                options_cirrus=[self.options[2]],
                adjusted=True,
            )
            return cloud_prob
        else:
            return None

    @property
    def cloud(self):
        """Calculate the cloud mask based on the cloud probability.

        Returns:
            2d array in bool: The cloud mask.
        """
        if self.activated:
            return self.prob_cloud > self.threshold
        return None
    
    @property
    def cold_cloud(self):
        """
        Determines if a pixel is classified as a extremly cold cloud.

        Returns:
            bool: True if the pixel is classified as a cold cloud, False otherwise.
        """
        if (self.activated and self.image.data.exist("tirs1")):
            return self.image.data.get("tirs1") < (self.surface_temp_low - self.threshold_cold_cloud) # in degree
        return None

    def set_options(self, options_var, options_temp, options_cirrus):
        """set the options for cloud probabilities

        Args:
            options_var (list of bool): variation options
            options_temp (list of bool): temporal options
            options_cirrus (list of bool): cirrus options
        """
        self.options_var = copy.deepcopy(options_var)
        self.options_temp = copy.deepcopy(options_temp)
        self.options_cirrus = copy.deepcopy(options_cirrus)

    def init_cloud_probability(self) -> None:
        """
        Generates the cloud probability layers based on the datacube.

        Returns:
            None
        """
        # create the probability layers according to the datacube
        (
            self.activated,
            self.pcp,
            self.lprob_var,
            self.lprob_temp,
            self.wprob_temp,
            self.wprob_bright,
            self.prob_cirrus,
            self.water,
            self.snow,
            self.surface_temp_low,  # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
            self.surface_temp_high,
        ) = compute_cloud_probability_layers(self.image, min_clear=self.min_clear)

    def select_cloud_probability(
        self,
        seed=None,
        label_cloud=1,
        label_noncloud=0,
        options_var=None,
        options_temp=None,
        options_cirrus=None,
        adjusted=True,
        show_figure=False,
    ):
        """Selects the cloud probability based on the given parlayer of seed with cloud and noncloud

        Args:
            seed (optional): The seed used for random number generation. Defaults to None.
            label_cloud (optional): The label for cloud pixels. Defaults to 1.
            label_noncloud (optional): The label for non-cloud pixels. Defaults to 0.
            options_var (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            options_temp (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            options_cirrus (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            adjusted (optional): The boolean to adjust the threshold. Defaults to True.

        Returns:
            mask_prob_final: The final mask probability.
            options: The selected options for variation, brightness, and cirrus.
            thrd_opt: The optimal threshold value.
        """
        # mask out the absolute clear pixels for land and water pixels
        mask_absclear_land = self.abs_clear_land
        mask_absclear_water = self.abs_clear_water

        # Test the optimal model by histogram overlapped
        # Convert as exporting cloud probability
        prob_record = None
        ol_record = 1.0
        options_record = None
        threshold_record = 0
        # at default, we use the current options
        if options_var is None:
            options_var = self.options_var
        if options_temp is None:
            options_temp = self.options_temp
        if options_cirrus is None:
            options_cirrus = self.options_cirrus

        # in case no cirrus prob. e.g., Landsat 4-7
        if self.woc == 0:
            options_cirrus = [False]

        bin_width = 0.025  # i.e., scale the cloud probability to 0-1 with 0.025
        for cir in options_cirrus:  # cirrus band has the highest priority
            for tmp in options_temp:  # at the second priority
                for var in options_var:  # variation and brightness prob.
                    # skip the option, when we do not use any components
                    if (not var) and (not tmp) and (not cir):
                        continue
                    # skip the option, only var is used
                    if (var) and (not tmp) and (not cir):
                        continue
                    # when the perfect model was created at the past round
                    if ol_record == 0:
                        continue

                    # check prob. over land that we have to use the previous prob. for land
                    mask_prob = combine_cloud_probability(
                        var,
                        tmp,
                        cir,
                        self.lprob_var,
                        self.lprob_temp,
                        self.prob_cirrus,
                        self.woc,
                        mask_absclear_land,
                        adjusted=adjusted,
                    )
                    # check prob. over water
                    if np.any(mask_absclear_water):
                        # in case when only temporial prob. is used (actually we do not have for water in Sentinel-2), we use the previous prob. for water
                        if (
                            (not var)
                            & (not cir)
                            & tmp
                            & (not isinstance(self.wprob_temp, np.ndarray))
                        ):
                            # pylint: disable=unsubscriptable-object
                            if prob_record is not None:
                                mask_prob = np.where(
                                    np.bitwise_and(self.water, prob_record > mask_prob),
                                    prob_record,
                                    mask_prob,
                                )
                        else:
                            mask_prob_water = combine_cloud_probability(
                                var,
                                tmp,
                                cir,
                                self.wprob_bright,
                                self.wprob_temp,
                                self.prob_cirrus,
                                self.woc,
                                mask_absclear_water,
                                adjusted=adjusted,
                            )
                            # mask_prob[self.water] = mask_prob_water[self.water] # update the cloud probability for water by replacing

                            # only update the pixels where the prob. over water is higher than the prob. over land
                            # which is the same as the previous Fmask, where the logistical math 'or' is used.
                            # to mask the thin cloud over the water just.
                            mask_prob = np.where(
                                np.bitwise_and(self.water, mask_prob_water > mask_prob),
                                mask_prob_water,
                                mask_prob,
                            )

                    if seed is not None:
                        # convert the cloud probability to the seed groups with 1 array
                        seed_cloud_prob, seed_noncloud_prob, prob_range = (
                            convert2seedgroups(
                                mask_prob,
                                seed,
                                label_cloud,
                                label_noncloud,
                                bin_width=bin_width,
                                equal_num=False,
                            )
                        )
                        # get the overlap rate between cloud and non-cloud pixels
                        ol, opt_thrd = overlap_cloud_probability(
                            seed_cloud_prob,
                            seed_noncloud_prob,
                            prob_range=prob_range,
                            prob_bin=bin_width,
                        )
                        if C.MSG_FULL:
                            print(
                                f">>> cloud probability ({str(var)[0]}{str(tmp)[0]}{str(cir)[0]}) | overlap: {ol:.9f} | optimal threshold: {opt_thrd:.9f}"
                            )
                        # ol, thrd_opt = overlap_cloud_probability(mask_prob, mask_seed, label_cloud=1, label_noncloud=0, prob_range = prob_range, prob_bin=0.025)
                        if show_figure:
                            utils.show_cloud_probability_hist(
                                seed_cloud_prob,
                                seed_noncloud_prob,
                                prob_range,
                                prob_bin=bin_width,
                                title=f"Cloud probability ({str(var)[0]}{str(tmp)[0]}{str(cir)[0]})",
                            )
                    else:
                        ol = 1.0 # 100% overlap between cloud and noncloud pixels
                        opt_thrd = self.threshold # default threshold
                    # update the optimal model if the overlap rate is decreased
                    if (ol_record == 1) or (
                        ((ol_record - ol) / (ol_record + C.EPS)) > self.overlap
                    ):  # 2 % decreased
                        ol_record = ol  # that cannot use .copy()
                        prob_record = mask_prob.copy()
                        options_record = [var, tmp, cir]
                        threshold_record = opt_thrd

        if (
            seed is not None
        ):  # only for the seed pixels which are used to find the optimal threshold
            self.options = options_record  # update the options determined
            self.threshold = threshold_record  # update the threshold determined
            if C.MSG_FULL:
                print(
                    f">>> optimal cloud probability ({str(options_record[0])[0]}{str(options_record[1])[0]}{str(options_record[2])[0]}) | optimal threshold: {threshold_record:.2f}"
                )

        return prob_record, options_record, threshold_record

    def match_cloud2shadow(
        self,
        cloud_objects,
        cloud_regions,
        pshadow,
    ):
        """
        Match shadows by identified clouds.

        Args:
            cloud_objects (ndarray): Binary cloud object mask.
            cloud_regions (list): List of cloud regions.
            pshadow (ndarray, number): Potential shadow layer.

        Returns:
            tuple: A tuple containing the projected cloud shadows and the updated cloud layer.
        """
        plane2image_row, plane2image_col, plane_offset = project_dem2plane(
            self.image.data.get("dem"),
            self.image.sun_elevation,
            self.image.sun_azimuth,
            self.image.resolution,
            self.image.filled,
        )
        sensor_zenith = self.image.read_angle("SENSOR_ZENITH", unit="degree")
        sensor_azimuth = self.image.read_angle("SENSOR_AZIMUTH", unit="degree")
        
        shadow_last, _ = match_cloud2shadow(
            cloud_regions,
            cloud_objects,
            pshadow,
            self.image.filled,
            sensor_zenith,
            sensor_azimuth,
            self.image.sun_elevation,
            self.image.sun_azimuth,
            self.image.resolution,
            similarity=self.similarity,
            sampling_cloud=self.sampling_cloud,
            thermal=self.image.data.get("tirs1"),
            surface_temp_low=self.surface_temp_low,
            surface_temp_high=self.surface_temp_high,
            ele=self.image.data.get("dem"),
            PLANE2IMAGE_ROW=plane2image_row,
            PLANE2IMAGE_COL=plane2image_col,
            PLANE_OFFSET=plane_offset,
            apcloud=False,
        )
        return shadow_last

    def __init__(self, predictors, woc, threshold, overlap=0.0) -> None:
        """Initialize the physical model"""
        # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
        self.image = None
        # the predictors for the physical model
        self.predictors = predictors
        # weight of cirrus probability
        self.woc = woc
        # threshold to separate cloud and non-cloud pixels
        self.threshold = threshold
        # the overlap density between cloud and non-cloud pixels to move further
        self.overlap = overlap  # 0% overlap increasing compared to the previous test to alter the physical models
        # extremely cold cloud
        self.threshold_cold_cloud = 35  # in degree
