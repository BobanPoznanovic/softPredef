from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model('mnist_model.h5');


def predict(input):

    a = model.predict_classes(input)

    return a