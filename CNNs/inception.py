import warnings
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


warnings.simplefilter(action="ignore", category=FutureWarning)

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = Flatten()(base_model.output)
x = Dense(4096, activation='relu')(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

BATCH_SIZE = 32
EPOCHS = 100

train_generator = train_datagen.flow_from_directory(
    'keras_test/dataset/train/',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    'keras_test/dataset/valid/',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='binary')

MODEL_FILE = 'filename.model'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.h5', save_best_only=True, monitor='val_accuracy', mode='min')

history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=320,
    validation_steps=200,
    verbose=1,
    callbacks=[earlyStopping, mcp_save]
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_weights.h5")


