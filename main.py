from camera import SignCamera
#from model import Model
import numpy as np

signInterpreter = SignCamera()
actions = np.array(["hello", "thanks",])


"""
Open The default camera with pose markings
"""
signInterpreter.open_camera_default()



"""
Open Camera for training
Actions that the code will try to detect
myModel = Model(actions)
signInterpreter.open_camera_and_save_data_for_training(actions)
myModel.train()
"""

"""
Open Camera for interpretation If the model has been trained
myModel = Model(actions)
myModel.load_trained_model()
signInterpreter.open_camera_and_predict(myModel,actions)
"""

