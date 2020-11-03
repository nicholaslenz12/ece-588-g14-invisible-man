from keras.models import load_model, model_from_json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input

with open('./model.json', 'r') as reader:
    arch = reader.read();
    
model = model_from_json(arch)
model.load_weights('./model_weights.h5')
#  print(model.summary())

opt = Adam(lr=0.1)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

eval_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
ev = eval_generator.flow_from_directory('keras_test/dataset/train/',
                                   class_mode='binary')

predictions = model.predict_generator(ev)
print(predictions)
