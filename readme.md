# Fmask
Fmask (Function of mask) is an automated algorithm for detecting clouds and cloud shadows in Landsat 4–9 (including 4, 5, 7, 8, and 9) and Sentinel-2 imagery. Version 5.0 introduces a Physics-Informed Machine Learning (PIML) framework (Figure 1) to enhance cloud detection accuracy, while cloud shadow detection relies on the physical geometric relationship between identified clouds and their corresponding shadows.

PIML flowchart

Figure 1: Flowchart of physics-informed machine learning (PIML) for cloud detection. The approach utilizes pixel-based LightGBM and CNN-based UNet models. The arrow indicates the processing sequence, transitioning from gray to black arrows. Abbreviations: HOT: Haze Optimized Transformation.

# Complete Package
This repository only provides the source code and does not include the integrated global auxiliary datasets or pre-trained machine learning models.

To access the complete Fmask packages, including all historical versions, you can download them from the Fmask Google Drive.

# How to Use
## Installation
TBD

## Running Fmask
To apply Fmask-UPL on a single Landsat 8-9 image:
```bash
python fmask.py --imagepath /path/to/image_directory_landsat8-9 --model UPL
```

To apply Fmask-UPL on a single Sentinel-2 image:
```bash
python fmask.py --imagepath /path/to/image_directory_Sentinel-2.SAFE --model UPL
```

To apply Fmask-LPL on a single Landsat 4-7 image:
```bash
python fmask.py --imagepath /path/to/image_directory_landsat4-7 --model LPL
```

## 🛠️ Command-Line Options
| Option             | Short | Description                                                                                     | Default |
|--------------------|-------|-------------------------------------------------------------------------------------------------|---------|
| `--imagepath`      | `-i`  | Path to input image directory (Landsat/Sentinel-2).                                             | *required* |
| `--model`          | `-m`  | Cloud detection model to use.                                                                   | `UPL` |
| `--dcloud`         | `-c`  | Dilation size (in pixels) for cloud mask.                                                       | `3`     |
| `--dshadow`        | `-s`  | Dilation size (in pixels) for cloud shadow mask.                                                | `5`     |
| `--dsnow`          | `-n`  | Dilation size (in pixels) for snow/ice mask.                                                    | `0`     |
| `--output`         | `-o`  | Directory for saving output. If not provided, results go into the input image directory.        | `None`  |
| `--skip_existing`  | `-s`  | Skip processing if results already exist (`yes` or `no`).                                       | `no`    |
| `--save_metadata`  | `-md` | Save model metadata as CSV.                                                                    | `no`    |
| `--display_fmask`  | `-df` | Save and display the Fmask result as a PNG.                                                     | `no`   |
| `--display_image`  | `-di` | Save and display the color composite PNG (NGR: NIR-Green-Red and SNG: SWIR1-NIR-Red).           | `no`   |
| `--print_summary`  | `-ps` | Print cloud, shadow, snow, and clear percentage summary.                                        | `no`    |

### Output
The tool generates a uint8 GeoTIFF file named after the selected cloud detection model. Each pixel is classified with one of the following values:

| Value | Class           | Description                                                                                                         |
|-------|---------------- |---------------------------------------------------------------------------------------------------------------------|
| 0     | Land            | Clear land surface                                                                                                  |
| 1     | Water           | Clear water surface                                                                                                 |
| 2     | Cloud Shadow    | Shadow matched with the detected cloud                                                                              |
| 3     | Snow/Ice        | Snow- or ice-covered surface                                                                                        |
| 4     | Cloud           | Detected cloud                                                                                                      |
| 255   | Filled          | No-data fill (e.g., due to missing input band(s))                                                                   |

> **Note:** Water and snow/ice pixels are labeled solely to enhance cloud detection. Their detection accuracy has not been evaluated.


### Version History
#### 5.0.0
- Applied Physics-Informed Machine Learning (PIML) framework for cloud detection, as described in Qiu et al., 2025.
- Adapted cloud shadow detection from MATLAB Fmask 4.6 with minor improvements described on [this page](https://github.com/qsly09/fmask5/wiki/Cloud-Shadow-Detection).

#### 1 - 4.6
Earlier versions of the Fmask tools offered only a physical-rule-based cloud detection module, programmed in MATLAB. See [this page](https://github.com/GERSL/Fmask) for more details.

## Contributing
We welcome and encourage contributions to Fmask! There are two primary ways to contribute:

## Report Issues or Suggestions
If you happen to have any issues or suggestions for improving Fmask, we encourage you to open an issue or submit a pull request.

## Share Problematic Images
We are actively collecting examples of images that have not been processed accurately by the current version of Fmask. If you come across such images, please share the image ID with us via this Google Sheet. The collected images will be used to refine the inner machine learning models, improving their accuracy and reliability in future versions.

## Known Issues
- False positive errors in cloud detection over bright surfaces. Although the most recent version of Fmask has addressed most of these issues, challenges remain in highly reflective areas, such as high-mountain snow and ice.
- Artifacts in cloud detection under very thin clouds. Thin (cirrus) clouds over bright surfaces, such as buildings and cropland, are more easily identified, as their features become more pronounced when the bright surfaces are located beneath very thin cirrus clouds.
- Potential omitted cloud shadows at the image boundary, where the associated clouds are either not identified or difficult to match outside the extent of the imagery (unable to detect beyond the image boundaries).
*Note*: Our team is collecting images with cloud detection issues and will continuously update the machine learning model to make improvements.

## References
Qiu, S., Zhu, Z., Yang, X., Ju, J., Zhou, Q., Neigh, C., Physics-Informed Machine Learning for Cloud Detection in Landsat and Sentinel-2 Imagery, Under review

Qiu, S., et al., Fmask 4.0: Improved cloud and cloud shadow detection in Landsats 4-8 and Sentinel-2 imagery, Remote Sensing of Environment, (2019), doi.org/10.1016/j.rse.2019.05.024 (paper for 4.0).

Zhu, Z. and Woodcock, C. E., Improvement and Expansion of the Fmask Algorithm: Cloud, Cloud Shadow, and Snow Detection for Landsats 4-7, 8, and Sentinel 2 images, Remote Sensing of Environment, (2014), doi:10.1016/j.rse.2014.12.014 (paper for version 3.2).

Zhu, Z. and Woodcock, C. E., Object-based cloud and cloud shadow detection in Landsat imagery, Remote Sensing of Environment, (2012), doi:10.1016/j.rse.2011.10.028 (paper for 1.6).

Qiu, S., et al., Improving Fmask cloud and cloud shadow detection in mountainous areas for Landsats 4–8 images, Remote Sensing of Environment, (2017), doi.org/10.1016/j.rse.2017.07.002 (paper for Mountainous Fmask (MFmask), which has been integrated into the current Fmask).

## Contact Us
Shi Qiu (shi.qiu@uconn.edu) and Zhe Zhu (zhe@uconn.edu)

Global Environmental Remote Sensing Laboratory (GERSL), University of Connecticut, Storrs, USA
