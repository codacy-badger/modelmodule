# -*- coding: utf-8 -*-
"""
This module loads scikit-learn pre-trained machine learning models and 
pipelines for production environments. Models are loaded using the
pickle module.

Any model is fit for loading, make sure the prediction input has the same 
number of features as in the training process.

This module does not deal with request queues and parallel processing. 
That should be managed by other module (web server etc).

Example usage:
    from modelmodule.MLModel import MLModel
    model = MLModel()
    result = model.predict(DATA)

DATA may be 1 to n objects

Todo:
    * apply PEP8
    * Apply PEP257
"""

import pickle
import os
import numpy as np


class MLModel:
    """
    Generic Wrapper class for loading any kind of scikit-learn model or pipeline.

    Attributes:
        model_path (str): full path of the model pickle file (.pkl)

    """

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
        elif(np.shape(features)[1] != self.model.n_features_):
            raise ValueError(f"input must have {self.model.n_features_} \
            features")

        result = self.model.predict(features)
        return result

    def load_newest_model(self):
        current_path = os.path.dirname(__file__)
        model_folder = 'models/' 
        model_folder = os.path.join(current_path, model_folder)

        os.chdir(model_folder)
        files = sorted(filter(os.path.isfile, os.listdir(model_folder)), 
            key=os.path.getmtime)
        files.reverse()
        model_path = files[0]

        with open(model_path, 'rb') as handle:
            self.model = pickle.load(handle)