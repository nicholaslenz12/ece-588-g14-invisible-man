from keras.models import load_model, model_from_json
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

with open('./model.json', 'r') as reader:
    arch = reader.read();
    
model = model_from_json(arch)
model.load_weights('./model_weights.h5')
#  print(model.summary())

opt = Adam(lr=0.1)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

eval_generator = ImageDataGenerator()
eval_generator.flow_from_directory('keras_test/dataset/train/',
                                   class_mode='binary')

loss = model.predict_generator(eval_generator)
