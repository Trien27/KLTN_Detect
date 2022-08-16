import tensorflow
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from get_data import pre_data
from plot_keras_history import plot_history
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


input_shape = (224, 224, 3)  
batch_size = 32


num_classes = 1  # number of classes in the data 0-1
epochs = 100

filepath="save_model_training/checkpoint/" + "_weights_improvement_{epoch:02d}_{val_accuracy:.2f}.h5"
checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss', verbose=1,
                                 save_best_only=False, save_weights_only=False, period=5)

# ngừng train model khi không có tiến bộ
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=75)

# khởi tạo model
base_model = MobileNetV2(input_shape=input_shape, alpha=1.0, include_top=False)
w = base_model.output
w = Flatten()(w)
w = Dense(128, activation="relu")(w)
output = Dense(num_classes, activation="sigmoid")(w)
model = Model(inputs=[base_model.input], outputs=[output])

model.compile(optimizer=Adam(lr=0.01), loss = 'binary_crossentropy', metrics = 'accuracy')
model.summary()


# Get generators for training and validation
train_generator, validation_generator = pre_data()


history = model.fit_generator(train_generator,
                            steps_per_epoch=64,
                            validation_data=validation_generator,
                            validation_steps=64,
                            epochs=epochs,
                            verbose = 1,  callbacks = [checkpoint, es])

plot_history(history)
plt.show()
plot_history(history, path="standard.png") 
plt.close()

# Save the model
model.save('my_model.h5')






