from dotenv import load_dotenv
load_dotenv("removewarnings.env")

import tensorflow as tf
import visualkeras

model = tf.keras.models.load_model("./model/model.keras")

for layer in model.layers:
    layer.output_shape = layer.output.shape

visualkeras.layered_view(model,  to_file="graph.png", sizing_mode='logarithmic', legend=True, show_dimension=True, max_xy=50)
