import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tensorflow.keras.layers import Dense, Lambda, Flatten, Conv2D, Input, MaxPooling2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config_ import dataset_path, folder_path
from glob import glob


class train(object):
    def __init__(self):
        # re-size all the images to this
        self.IMAGE_SIZE = [224, 224]

        # specify train and valid path
        self.train_path = dataset_path + 'Train/'
        self.valid_path = dataset_path + 'Test/'
        self.folders = glob(self.train_path + '*')

    def model(self):
        # importing vgg19 model
        model = VGG19(input_shape=self.IMAGE_SIZE + [3], weights='imagenet', include_top=False)

        # freezing the layers
        for layers in model.layers:
            layers.trainable = False

        x = Flatten()(model.output)

        prediction = Dense(len(self.folders), activation='softmax')(x)
        # create a model object
        model = Model(inputs=model.input, outputs=prediction)

        # tell the model what cost and optimization method to use
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        return model

    def run_model(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Make sure you provide the same target size as initialized for the image size
        training_set = train_datagen.flow_from_directory(self.train_path,
                                                         target_size=(224, 224),
                                                         batch_size=32,
                                                         class_mode='categorical')

        test_set = test_datagen.flow_from_directory(self.valid_path,
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='categorical')
        print("[Info] -- model started training --")
        model_ = self.model()
        r = model_.fit_generator(
            training_set,
            validation_data=test_set,
            epochs=50,
            steps_per_epoch=len(training_set),
            validation_steps=len(test_set)
        )
        print("[Info] -- model trained successfully")

        model_.save(folder_path + './Models/model_vgg19.h5')
        print("[Info] -- model saved successfully --")


if __name__ == '__main__':
    Obj = train()
    Obj.run_model()
