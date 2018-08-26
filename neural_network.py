from keras.models import load_model

model = load_model('mnist_model.h5');


def predict(input):

    a = model.predict_classes(input)

    return a