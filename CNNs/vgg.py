import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten


warnings.simplefilter(action="ignore", category=FutureWarning)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.get_layer('block5_pool').output
x = Flatten(input_shape=base_model.output_shape[1:])(x)
x = Dense(2048, activation="relu")(x)
x = Dense(512, activation="relu")(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[:5]:
    layer.trainable = False

opt = Adam(lr=0.01)
model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator()

validation_datagen = ImageDataGenerator()

BATCH_SIZE = 16
EPOCHS = 100

train_generator = train_datagen.flow_from_directory(
    'keras_test/dataset/train/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'keras_test/dataset/valid/',
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='binary')

MODEL_FILE = 'vgg16_test.model'

# print(model.summary())

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.h5', save_best_only=True, monitor='val_accuracy', mode='min')

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[earlyStopping, mcp_save]
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.h5")


