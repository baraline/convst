# -*- coding: utf-8 -*-

__all__ = [
    "load_sktime_dataset_split",
    "load_sktime_dataset",
    "load_sktime_arff_file_resample_id",
    "load_sktime_arff_file",
    "return_all_dataset_names",
    "load_sktime_ts_file",
    "UCR_stratified_resample",
    "stratified_resample"
]
import numpy as np
from sktime.datasets.base import load_UCR_UEA_dataset
from sklearn.preprocessing import LabelEncoder
from sktime.utils.data_processing import from_nested_to_3d_numpy
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_io import load_from_arff_to_dataframe

from sklearn.utils import resample
class stratified_resample:
    def __init__(self, n_splits, n_samples_train):
        self.n_splits=n_splits
        self.n_samples_train=n_samples_train
        
    def split(self, X, y=None, groups=None):
        idx_X = np.asarray(range(X.shape[0]))
        for i in range(self.n_splits):
            if i == 0:
                idx_train = np.asarray(range(self.n_samples_train))
            else:
                idx_train = resample(idx_X, n_samples=self.n_samples_train, replace=False, random_state=i, stratify=y)
            idx_test = np.asarray(list(set(idx_X) - set(idx_train)))
            yield idx_train, idx_test
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class UCR_stratified_resample:
    def __init__(self, n_splits, path):
        self.n_splits=n_splits
        self.path=path
    
    def split(self, X, y=None, groups=None):
        for i in range(self.n_splits):
            X_train, X_test, y_train, y_test, _ = load_sktime_arff_file_resample_id(self.path,i)
            idx_Train = [np.where((X == X_train[j]).all(axis=2))[0][0] for j in range(X_train.shape[0])]
            idx_Test = [np.where((X == X_test[j]).all(axis=2))[0][0] for j in range(X_test.shape[0])]
            yield idx_Train, idx_Test
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

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


def load_sktime_arff_file_resample_id(path, rs_id, normalize=True):
    #Load datasets
    X_train, y_train = load_from_arff_to_dataframe(path+'_{}_TRAIN.arff'.format(rs_id))
    X_test, y_test = load_from_arff_to_dataframe(path+'_{}_TEST.arff'.format(rs_id))

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
