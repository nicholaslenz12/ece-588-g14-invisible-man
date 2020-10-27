import warnings
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

warnings.simplefilter(action="ignore", category=FutureWarning)

base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.4)(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=False)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=False)

BATCH_SIZE = 32
EPOCHS = 100

train_generator = train_datagen.flow_from_directory(
    'keras_test/dataset/train/',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    'keras_test/dataset/valid/',
    target_size=(299, 299),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

MODEL_FILE = 'filename.model'

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


