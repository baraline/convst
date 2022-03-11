.. _api:

=================
API Documentation
=================

Full API documentation of the *convst* Python package.

:mod:`convst.classifiers`: Classification algorithms
=====================================================

.. automodule:: convst.transformers
    :no-members:
    :no-inherited-members:

.. currentmodule:: convst

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   classifiers.R_DST_Ridge


:mod:`convst.transformers`: Transformation algorithms
=====================================================

.. automodule:: convst.transformers
    :no-members:
    :no-inherited-members:

.. currentmodule:: convst

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   transformers.R_DST

:mod:`convst.utils`: Utilities
==============================

.. automodule:: convst.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: convst

.. autosummary::
  :nosignatures:
  :toctree: generated/
  :template: class.rst
  
  utils.dataset_utils.UCR_stratified_resample
  utils.dataset_utils.stratified_resample
  
  :template: function.rst

  utils.shapelet_utils.generate_strides_2D
  utils.shapelet_utils.generate_strides_1D
  utils.check_utils.check_array_3D
  utils.check_utils.check_array_2D
  utils.check_utils.check_array_1D
  utils.check_utils.check_is_numpy_or_pd
  utils.check_utils.check_is_numpy
  utils.dataset_utils.load_sktime_dataset_split
  utils.dataset_utils.load_sktime_dataset
  utils.dataset_utils.load_sktime_arff_file_resample_id
  utils.dataset_utils.load_sktime_arff_file
  utils.dataset_utils.return_all_dataset_names
  utils.dataset_utils.load_sktime_ts_file
  