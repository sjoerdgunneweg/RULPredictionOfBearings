import numpy as np
import pandas as pd
from scipy.signal import hilbert
import os

class FeatureExtraction:
    def __init__(self, datasetPath:str):
        self.datasetPath = datasetPath
        self.dataset = self.readData()

    # Reads csv Datasets
    def readData(self) -> pd.DataFrame:
        return pd.read_csv(self.datasetPath, sep='\t')

    # Calculates Root Mean Square error
    def calculateRMS(self) -> list:
        return [np.sqrt(np.sum(np.square(self.dataset[column])) / len(self.dataset[column])) for column in self.dataset]
    
    # Calculates the Hilbert Huang Transform
    def calculateHHT(self):
        return hilbert(self.dataset)
    
    # Plots the Features
    def plotFeatures(self, data: pd.DataFrame):
        pass
    
    def reshapeAndConcat(self, rms:list, fileName:str, rmsFrame:pd.DataFrame) -> pd.DataFrame:
        rms = np.array(rms)
        rms = pd.DataFrame(rms.reshape(1, 4))
        rms.index = [fileName]
        rmsFrame = pd.concat([rmsFrame, rms])

        return rmsFrame