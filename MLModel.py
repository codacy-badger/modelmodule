import pickle
import os
import numpy as np

class MLModel:

    def __init__(self,model_path=None):
        ''' Constructor '''
        if(model_path is None):
            self.load_newest_model()
        else:
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

    def load_newest_model(self):
        current_path = os.path.dirname(__file__)
        model_folder = 'models/' 
        model_folder = os.path.join(current_path, model_folder)

        os.chdir(model_folder)
        files = sorted(filter(os.path.isfile, os.listdir(model_folder)), key=os.path.getmtime)
        files.reverse()

        model_path = files[0]

        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle)