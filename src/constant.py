"""constant.py: Constants for the Fmask5, whic"""

# The smallest positive float in order to avoid division by zero
# float16: 0.000977
# float32: 1.19209e-07
# float64: 2.220446049250313e-16
EPS = 1.19209e-07
LABEL_CLOUD = 4
LABEL_SNOW = 3
LABEL_SHADOW = 2
LABEL_WATER = 1
LABEL_LAND = 0
LABEL_CLEAR = 0
LABEL_FILL = 255
RANDOM_SEED = 42  # None: dynamic ||| 42 static seed for reproductivity A popular value chosen all over the world for the random state is 42. The reason was a little surprising and quirky. The number simply has been made popular by a comic science fiction called "The Hitchhiker's Guide to the Galaxy" authored by Douglas Adams in 1978
HIGH_LEVEL, LOW_LEVEL = 82.5, 17.5  # percentiles see Zhu and Curtis, 2012

NORTH_LAT_GSWO, SOUTH_LAT_GSWO = (
    78,
    -59,
)  # The north and south border of the GSWO, by examining the 30-m product. Any pixels within this range, the tool will not use the GSWO layer
MSG_BASE = True  # whether to show the base message
MSG_FULL = True  # whether to show the all message