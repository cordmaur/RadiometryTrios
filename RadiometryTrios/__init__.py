__version__ = '0.0.1'
import logging

from .trios_mdb import TriosMDB
from .radiometry import Radiometry, BaseRadiometry
from .radiometry_group import RadiometryGroup
from .radiometry_db import RadiometryDB

# Configure the logger for the whole package
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.WARNING, datefmt='%I:%M:%S')

