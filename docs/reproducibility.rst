.. _reproducibility:

===============
Reproducibility
===============

**convst** provides algorithms for time series classification that have
been published in the literature. To allow user to verify those results,
we provide instructions and scripts to generate the same experimental setup
that was used to obtain the results.


Obtain the UEA & UCR Datasets
-----------------------------

The `UEA & UCR Time Series Classification Repository <http://www.timeseriesclassification.com>`_
is an ongoing project to develop a comprehensive repository for research into
time series classification providing datasets as well as code and results for
many algorithms.

Convenience functions are provided in convst to download a dataset from this
repository by simply specifying its name:

* Original Train & Test splits for a dataset: :func:`convts.utils.load_sktime_dataset_split`,
* Full dataset: :func:`convts.utils.load_sktime_dataset`.

Obtain the UEA & UCR Resamples
------------------------------

On the `UEA & UCR Time Series Classification Repository <http://www.timeseriesclassification.com>`_
you can also find published results, using a 30 resamples validation. It is useful to use the same 
resamples to be able to directly compare yourself to the state of the art algorithms.

The published results are obtained using the `tsml java implementation <https://github.com/uea-machine-learning/tsml>`_, and the function used
to generate the resample is `resampleTrainAndTestInstances <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/utilities/InstanceTools.java#L181>`_ .
What you want to do is to download tsml and use the `DataHandling <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/examples/DataHandling.java>`_ 
example to generate the resamples from the Train and Test arff files for each datasets that you previously downloaded from `the UCR archive <http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip>`_ .

We provide two classes that can be used in sklearn cross validation tools, note that if using the
random one, the comparaison to published results will not be valid, but you won't need to perform the previous steps with tsml.

* Using resamples from tsml : :func:`convts.utils.UCR_stratified_resample`,
* Using random resamples : :func:`convts.utils.stratified_resample`.

Running the cross validation script
-----------------------------------

WIP