from tensorflow.keras.applications import MobileNetV3Small,DenseNet121
from tensorflow.keras.layers import Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import time

base_model = MobileNetV3Small(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
#base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x) 
x = Dropout(0.5)(x)
predictions = Dense(101, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.load_weights("mobilenet_v3.h5")
#model.load_weights("densenet.h5")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory="./test",
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False
)
start_time = time.time()
predictions = model.predict(test_generator)
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.2f} s".format(execution_time))
predicted_classes = np.argmax(predictions, axis=1)

class_names_df = pd.read_csv('tw_food_101_classes.csv', header=None)
class_names = class_names_df[1].tolist() 
predicted_labels = [class_names[idx] for idx in predicted_classes]

print(predicted_labels)