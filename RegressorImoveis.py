import pickle
import os
import numpy as np

class RegressorImoveis:

    def __init__(self,model_path=None):
        ''' Constructor '''
        if(model_path is None):
            current_path = os.path.dirname(__file__)
            rel_path = 'models/imoveis_df_random_forest_regressor.pkl' 
            model_path = os.path.join(current_path, rel_path)

        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle)

    def info(self):
        print(self.model)

    def predict(self,features=None):
        if(features is None ):
            raise ValueError("input must have at least 1 object")
        elif(np.shape(features)[1] != 4):
            raise ValueError("input must have 4 features")

        result = self.model.predict(features)

        return result