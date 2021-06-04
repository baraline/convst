"""Utilities.
"""

from .checks_utils import check_array_3D, check_array_2D, check_array_1D, check_is_numpy_or_pd, check_is_numpy

from .dataset_utils import load_sktime_dataset_split, load_sktime_dataset, load_sktime_arff_file_resample_id, stratified_resample
from .dataset_utils import load_sktime_arff_file, return_all_dataset_names, load_sktime_ts_file, UCR_stratified_resample

from .kernel_utils import apply_one_kernel_all_sample, apply_one_kernel_one_sample

from .shapelets_utils import generate_strides_2D, generate_strides_1D

__author__ = 'Antoine Guillaume antoine.guillaume45@gmail.com'

__all__ = ["generate_strides_2D","generate_strides_1D", "check_array_3D", "check_array_2D", 
"check_array_1D", "check_is_numpy_or_pd", "check_is_numpy","load_sktime_dataset_split",
"load_sktime_dataset","load_sktime_arff_file_resample_id",
"load_sktime_arff_file","return_all_dataset_names","load_sktime_ts_file",
"UCR_stratified_resample","stratified_resample","apply_one_kernel_one_sample",
"apply_one_kernel_all_sample"]