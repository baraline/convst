# -*- coding: utf-8 -*-

import numpy as np

from aeon.datasets._data_loaders import _load_dataset

from sklearn.preprocessing import LabelEncoder
from numba import njit, prange

from convst import __USE_NUMBA_CACHE__

@njit(cache=__USE_NUMBA_CACHE__)
def z_norm_3D(X):
    """
    Z normalise a time series dataset assumed to be of even length. A small value
    is added to the standard deviation for all samples and features to avoid
    0 division.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Input numerical array to z-normalise

    Returns
    -------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Z-normalised array

    """
    for i_x in prange(X.shape[0]):
        for i_ft in prange(X.shape[1]):
            X[i_x, i_ft] = (X[i_x, i_ft] - X[i_x, i_ft].mean())/(X[i_x, i_ft].std() + 1e-8)
    return X
    
def z_norm_3D_list(X):
    """
    Z normalise a time series dataset assumed to be of even length. A small value
    is added to the standard deviation for all samples and features to avoid
    0 division.

    Parameters
    ----------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Input numerical array to z-normalise

    Returns
    -------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Z-normalised array

    """
    for i_x in range(len(X)):
        for i_ft in range(len(X[i_x])):
            X[i_x][i_ft] = (X[i_x][i_ft] - X[i_x][i_ft].mean())/(X[i_x][i_ft].std() + 1e-8)
    return X
    

def load_UCR_UEA_dataset_split(name, normalize=False):
    """
    Load the original train and test splits of a dataset 
    from the UCR/UEA archive by name using aeon API.

    Parameters
    ----------
    name : string
        Name of the dataset to download.
    normalize : boolean, optional
        If True, time series will be z-normalized. The default is True.


    Returns
    -------
    X_train : array, shape=(n_samples_train, n_features, n_timestamps)
        Training data from the dataset specified by path.
    X_test : array, shape=(n_samples_test, n_features, n_timestamps)
        Testing data from the dataset specified by path.
    y_train : array, shape=(n_samples_train)
        Class of the training data.
    y_test : array, shape=(n_samples_test)
        Class of the testing data.
    le : LabelEncoder
        LabelEncoder object used to uniformize the class labels


    """
    #Load datasets
    X_train, y_train = _load_dataset(name, return_X_y=True, split='train')
    X_test, y_test = _load_dataset(name, return_X_y=True, split='test')

    #Convert class labels to make sure they are between 0,n_classes
    le = LabelEncoder().fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    min_len = min(
        min([len(X_train[i][0])for i in range(len(X_train))]),
        min([len(X_test[i][0])for i in range(len(X_test))]),
    )
    #Z-Normalize the data
    if normalize and not isinstance(X_train, list):
        X_train = z_norm_3D(X_train)
        X_test = z_norm_3D(X_test)
    if normalize and isinstance(X_train, list):
        X_train = z_norm_3D_list(X_train)
        X_test = z_norm_3D_list(X_test)
    return X_train, X_test, y_train, y_test, min_len


def load_UCR_UEA_dataset(name, normalize=False):
    """
    Load a dataset from the UCR/UEA archive by name using aeon API

    Parameters
    ----------
    name : string
        Name of the dataset to download.
    normalize : boolean, optional
        If True, time series will be z-normalized. The default is True.

    Returns
    -------
    X : array, shape=(n_samples, n_features, n_timestamps)
        Time series data from the dataset specified by name.
    y : array, shape=(n_samples)
        Class of the time series
    """
    #Load datasets
    X_train, X_test, y_train, y_test, _ = load_UCR_UEA_dataset_split(
        name, split='train', normalize=normalize
    )
    return np.concatenate((X_train, X_test),axis=0), np.concatenate((y_train, y_test),axis=0)

def return_all_dataset_names():
    return np.concatenate((
        return_all_univariate_dataset_names(),
        return_all_multivariate_dataset_names(),
        return_all_variable_univariate_dataset_names(),
        return_all_variable_multivariate_dataset_names()
    ))

def return_all_multivariate_dataset_names():
    
    return np.asarray([
        "ArticularyWordRecognition",
        "AtrialFibrillation",
        "BasicMotions",
        "Cricket",
        "DuckDuckGeese",
        "EigenWorms",
        "Epilepsy",
        "EthanolConcentration",
        "ERing",
        "FaceDetection",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "Libras",
        "LSST",
        "MotorImagery",
        "NATOPS",
        "PenDigits",
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "StandWalkJump",
        "UWaveGestureLibrary"
    ])

def return_all_variable_multivariate_dataset_names():
    return np.asarray([
        "AsphaltObstaclesCoordinates",
        "AsphaltPavementTypeCoordinates",
        "AsphaltRegularityCoordinates",
        "CharacterTrajectories",
        "InsectWingbeat",
        "JapaneseVowels",
        "SpokenArabicDigits"
    ])


def return_all_variable_univariate_dataset_names():
    return np.asarray([
        "AllGestureWiimoteX",
        "AllGestureWiimoteY",
        "AllGestureWiimoteZ",
        "GestureMidAirD1",
        "GestureMidAirD2",
        "GestureMidAirD3",
        "GesturePebbleZ1",
        "GesturePebbleZ2",
        "PickupGestureWiimoteZ",
        "PLAID",
        "ShakeGestureWiimoteZ"
    ])


def return_all_univariate_dataset_names():
    """
    Return the names of the 112 univariate datasets of the UCR archive.

    Returns
    -------
    array, shape=(112)
        Names of the univariate UCR datasets.

    """
    return np.asarray([
            "ACSF1",
            "Adiac",
            "ArrowHead",
            "Beef",
            "BeetleFly",
            "BirdChicken",
            "BME",
            "Car",
            "CBF",
            "Chinatown",
            "ChlorineConcentration",
            "Coffee",
            "Computers",
            "CricketX",
            "CricketY",
            "CricketZ",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW",
            "Earthquakes",
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "ElectricDevices",
            "EOGHorizontalSignal",
            "EOGVerticalSignal",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "FiftyWords",
            "Fish",
            "FordA",
            "FordB",
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "GunPoint",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",
            "Haptics",
            "Herring",
            "HouseTwenty",
            "InlineSkate",
            "InsectEPGRegularTrain",
            "InsectEPGSmallTrain",
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
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
            "MoteStrain",
            "NonInvasiveFetalECGThorax1",
            "NonInvasiveFetalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "Plane",
            "PowerCons",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxTW",
            "RefrigerationDevices",
            "ScreenType",
            "SemgHandGenderCh2",
            "SemgHandMovementCh2",
            "SemgHandSubjectCh2",
            "ShapeletSim",
            "ShapesAll",
            "SmallKitchenAppliances",
            "SmoothSubspace",
            "SonyAIBORobotSurface1",
            "SonyAIBORobotSurface2",
            "Strawberry",
            "SwedishLeaf",
            "Symbols",
            "SyntheticControl",
            "ToeSegmentation1",
            "ToeSegmentation2",
            "Trace",
            "TwoLeadECG",
            "TwoPatterns",
            "UMD",
            "UWaveGestureLibraryAll",
            "UWaveGestureLibraryX",
            "UWaveGestureLibraryY",
            "UWaveGestureLibraryZ",
            "Wafer",
            "Wine",
            "WordSynonyms",
            "Worms",
            "WormsTwoClass",
            "Yoga",
            "Phoneme",
            "PigAirwayPressure",
            "PigArtPressure",
            "PigCVP",
            "Crop",
            "StarLightCurves",
            "Rock",
            "HandOutlines",
            "CinCECGTorso",
            "EthanolLevel"
            ])
