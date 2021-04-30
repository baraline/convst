# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:46:14 2021

@author: A694772
"""
__all__ = [
	"load_sktime_dataset_split",
    "load_sktime_dataset",
    "return_all_dataset_names"
]
import numpy as np
from sktime.datasets.base import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder
from sktime.utils.data_processing import from_nested_to_3d_numpy


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
    return ["ACSF1",
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
            "CinCECGTorso",
            "Coffee",
            "Computers",
            "CricketX",
            "CricketY",
            "CricketZ",
            "Crop",
            "DiatomSizeReduction",
            "DistalPhalanxOutlineAgeGroup",
            "DistalPhalanxOutlineCorrect",
            "DistalPhalanxTW",
            "DodgerLoopDay",
            "DodgerLoopGame",
            "DodgerLoopWeekend",
            "Earthquakes",
            "ECG200",
            "ECG5000",
            "ECGFiveDays",
            "ElectricDevices",
            "EOGHorizontalSignal",
            "EOGVerticalSignal",
            "EthanolLevel",
            "FaceAll",
            "FaceFour",
            "FacesUCR",
            "FiftyWords",
            "Fish",
            "FordA",
            "FordB",
            "FreezerRegularTrain",
            "FreezerSmallTrain",
            "Fungi",
            "GestureMidAirD1",
            "GestureMidAirD2",
            "GestureMidAirD3",
            "GunPoint",
            "GunPointAgeSpan",
            "GunPointMaleVersusFemale",
            "GunPointOldVersusYoung",
            "Ham",
            "HandOutlines",
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
            "MelbournePedestrian",
            "MiddlePhalanxOutlineAgeGroup",
            "MiddlePhalanxOutlineCorrect",
            "MiddlePhalanxTW",
            "MixedShapesRegularTrain",
            "MixedShapesSmallTrain",
            "MoteStrain",
            "NonInvasiveFatalECGThorax1",
            "NonInvasiveFatalECGThorax2",
            "OliveOil",
            "OSULeaf",
            "PhalangesOutlinesCorrect",
            "Phoneme",
            "PigAirwayPressure",
            "PigArtPressure",
            "PigCVP",
            "Plane",
            "PowerCons",
            "ProximalPhalanxOutlineAgeGroup",
            "ProximalPhalanxOutlineCorrect",
            "ProximalPhalanxTW",
            "RefrigerationDevices",
            "Rock",
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
            "StarLightCurves",
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
            "Yoga"]