import tensorflow as tf
from keras_visualizer import visualizer

my_settings = {
    'MAX_NEURONS': None,
    'INPUT_DENSE_COLOR': 'teal',
    'HIDDEN_DENSE_COLOR': 'gray',
    'OUTPUT_DENSE_COLOR': 'crimson'
}

model = tf.keras.models.load_model("./model/model.keras")

visualizer(model, file_format='png', settings=my_settings)
