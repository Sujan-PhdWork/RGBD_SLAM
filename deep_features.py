import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt


model = keras.applications.VGG16(weights='imagenet', include_top=True)
def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

img, x = load_image("demo.jpg")
feat = feat_extractor.predict(x)

plt.figure(figsize=(16,4))
plt.plot(feat[0])
plt.show()
