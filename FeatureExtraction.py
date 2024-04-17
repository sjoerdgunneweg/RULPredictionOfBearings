import numpy as np
import pandas as pd
from scipy.signal import hilbert

class FeatureExtraction:
    def __init__(self, datasetPath:str):
        self.datasetPath = datasetPath
        self.dataset = self.readData()

    # Reads csv Datasets
    def readData(self) -> pd.DataFrame:
        dataset = pd.read_csv(self.datasetPath, sep='\t')
        # nColumns = len(dataset.columns)

        # if (nColumns== 4):
        #     dataset.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']
        # elif (nColumns == 8):
        #     dataset.columns = ['Bearing 1','Bearing 1','Bearing 2','Bearing 2', 'Bearing 3','Bearing 3','Bearing 4','Bearing 4']

        return dataset
    
    # Calculates Root Mean Square error
    def calculateRMS(self) -> list:
        return [np.sqrt(np.sum(np.square(self.dataset[column])) / len(self.dataset[column])) for column in self.dataset]
    
    # Calculates the Hilbert Huang Transform
    def calculateHHT(self):
        return hilbert(self.dataset)