import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image


class test(object):
    def __init__(self, test_file):

        self.model_path = './Models/model_vgg19.h5'
        self.test_file = test_file

    def load_model(self):

        model = load_model(self.model_path)

        return model

    def run(self):

        img = image.load_img(self.test_file,
                             target_size=(224, 224))
        x = image.img_to_array(img)
        x = x / 255
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        model_ = self.load_model()
        prediction = np.argmax(model_.predict(img_data), axis=1)

        """
        {'Falling': 0, 'WithoutFalling': 1}
        """

        if prediction[0] == 0:
            print("Falling")
        else:
            print("No falling")


if __name__ == '__main__':
    image_path = ''
    Obj = test(image_path)
    Obj.run()
