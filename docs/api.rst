.. _api:

=================
API Documentation
=================

Full API documentation of the *convst* Python package.

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

   transformers.MiniRocket
   transformers.Convolutional_shapelet
   transformers.kernel
   transformers.Rocket_kernel
   transformers.MiniRocket_kernel
   transformers.ConvolutionalShapeletTransformer

:mod:`convst.interpreters`: Interpretability tools
==================================================

.. automodule:: convst.interpreters
    :no-members:
    :no-inherited-members:

.. currentmodule:: convst

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   interpreters.CST_interpreter


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
  
  utils.UCR_stratified_resample
  utils.stratified_resample
  
  :template: function.rst

  utils.generate_strides_2D
  utils.generate_strides_1D
  utils.check_array_3D
  utils.check_array_2D
  utils.check_array_1D
  utils.check_is_numpy_or_pd
  utils.check_is_numpy
  utils.load_sktime_dataset_split
  utils.load_sktime_dataset
  utils.load_sktime_arff_file_resample_id
  utils.load_sktime_arff_file
  utils.return_all_dataset_names
  utils.load_sktime_ts_file
  utils.apply_one_kernel_one_sample
  utils.apply_one_kernel_all_sample
