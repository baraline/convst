# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
    "load_sktime_dataset_split",
    "load_sktime_dataset",
    "return_all_dataset_names",
    "load_sktime_ts_file"
]
import numpy as np
from sktime.datasets.base import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_arff_to_dataframe

def load_sktime_dataset_split(name, normalize=True):
    #Load datasets
    X_train, y_train = load_UCR_UEA_dataset(name, return_X_y=True, split='train')
    X_test, y_test = load_UCR_UEA_dataset(name, return_X_y=True, split='test')

    #Convert pandas DataFrames to numpy arrays
    X_train = from_nested_to_3d_numpy(X_train)
    X_test = from_nested_to_3d_numpy(X_test)

    #Convert class labels to make sure they are between 0,n_classes
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    #Z-Normalize the data
    if normalize:
        X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / (
            X_train.std(axis=-1, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=-1, keepdims=True)) / (
            X_test.std(axis=-1, keepdims=True) + 1e-8)

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test, le


def load_sktime_arff_file(path, normalize=True):
    #Load datasets
    X_train, y_train = load_from_arff_to_dataframe(path+'_TRAIN.arff')
    X_test, y_test = load_from_arff_to_dataframe(path+'_TEST.arff')

    #Convert pandas DataFrames to numpy arrays
    X_train = from_nested_to_3d_numpy(X_train)
    X_test = from_nested_to_3d_numpy(X_test)

    #Convert class labels to make sure they are between 0,n_classes
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    #Z-Normalize the data
    if normalize:
        X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / (
            X_train.std(axis=-1, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=-1, keepdims=True)) / (
            X_test.std(axis=-1, keepdims=True) + 1e-8)

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test, le

def load_sktime_ts_file(path, normalize=True):
    #Load datasets
    X_train, y_train = load_from_tsfile_to_dataframe(path+'_TRAIN.ts')
    X_test, y_test = load_from_tsfile_to_dataframe(path+'_TEST.ts')

    #Convert pandas DataFrames to numpy arrays
    X_train = from_nested_to_3d_numpy(X_train)
    X_test = from_nested_to_3d_numpy(X_test)

    #Convert class labels to make sure they are between 0,n_classes
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    #Z-Normalize the data
    if normalize:
        X_train = (X_train - X_train.mean(axis=-1, keepdims=True)) / (
            X_train.std(axis=-1, keepdims=True) + 1e-8)
        X_test = (X_test - X_test.mean(axis=-1, keepdims=True)) / (
            X_test.std(axis=-1, keepdims=True) + 1e-8)

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test, le

def load_sktime_dataset(name, normalize=True):
    #Load datasets
    X, y = load_UCR_UEA_dataset(name, return_X_y=True)

    #Convert pandas DataFrames to numpy arrays
    X = from_nested_to_3d_numpy(X)

    #Convert class labels to make sure they are between 0,n_classes
    le = LabelEncoder().fit(y)
    y = le.transform(y)

    #Z-Normalize the data
    if normalize:
        X = (X - X.mean(axis=-1, keepdims=True)) / (
            X.std(axis=-1, keepdims=True) + 1e-8)

    return X.astype(np.float32), y, le


def return_all_dataset_names():
    return ["Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "CBF",
    "Car",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "Earthquakes",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFatalECGThorax1",
    "NonInvasiveFatalECGThorax2",
    "OSULeaf",
    "OliveOil",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarlightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",]
