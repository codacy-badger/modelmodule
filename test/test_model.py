import pytest

from ..MLModel import MLModel
import random

# Provides wrong path and checks that the module is not loading any default or newest model
def test_load_model_with_wrong_path():
    with pytest.raises(Exception):
        model = MLModel('WRONG PATH')

# Provides right path (tester dependant) and asserts if module could load the model
def test_load_model_with_right_path():
    model = MLModel('/home/user/proj/modelmodule/models/random_forest_regressor.pkl')
    assert model is not None

# No model path provided (default fallback gets base pretrained model)
def test_load_model_without_path():
    model = MLModel()
    assert model is not None

# Model loaded, no objects in the predict input
def test_predict_without_input():
    model = MLModel()
    with pytest.raises(ValueError):
        model.predict(None)

lat = -15.738213539123535
lng = -47.897647857666015
area = 89
rooms = 3.0

# input provided is 1 object, wrong col number
def test_predict_without_input():
    model = MLModel()
    with pytest.raises(ValueError):
        model.predict([[area,rooms,lat]])

# 1 object, no problems
def test_predict_one_object_no_problems():
    apt_316_n = [[area,rooms,lat,lng]]
    model = MLModel()
    result = model.predict(apt_316_n)
    assert result is not None

# n objects, no problems
def test_predict_n_objects_no_problems():
    model = MLModel()
    apto = [area,rooms,lat,lng]
    aptos = []
    for i in range(random.randint(1,10)):
        aptos.append(apto)
    print("\n test_predict_n_objects: total of {} apts".format(len(aptos)))
    result = model.predict(aptos)
    print(result)
    assert (result is not None) & (len(result)>0)