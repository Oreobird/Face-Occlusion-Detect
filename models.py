import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import heapq

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

# np.set_printoptions(threshold=np.nan)

EPOCHS = 25

class FodNet:
    def __init__(self, dataset, class_num, batch_size, input_size, fine_tune=True, fine_tune_model_file='imagenet'):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.dataset = dataset
        self.fine_tune_model_file = fine_tune_model_file
        if fine_tune:
            self.model = self.fine_tune_model()
        else:
            self.model = self.__create_model()
            
    def __base_model(self, inputs):
        
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(feature)
        
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(feature)

        return feature
    
    def __dense(self, feature):
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dense(units=128)(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=256)(feature)
        feature = tf.keras.layers.BatchNormalization()(feature)
        feature = tf.keras.layers.Activation(activation=tf.nn.relu)(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        return feature
    
    def __create_model(self):
        input_fod = tf.keras.layers.Input(name='fod_input', shape=(self.input_size, self.input_size, 1))

        feature_fod = self.__base_model(input_fod)
        feature_fod = self.__dense(feature_fod)

        output_fod = tf.keras.layers.Dense(name='fod_output', units=self.class_num, activation=tf.nn.sigmoid)(feature_fod)

        model = tf.keras.Model(inputs=[input_fod], outputs=[output_fod])

        losses = {
            'fod_output': 'binary_crossentropy',
        }

        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=losses,
                      metrics=['accuracy'])

        return model

    def __extract_output(self, model, name, input):
        model._name = name
        for layer in model.layers:
            layer.trainable = True
        return model(input)
    
    def fine_tune_model(self):
        input_fod = tf.keras.layers.Input(name='fod_input', shape=(self.input_size, self.input_size, 3))

        # resnet_fod = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        # feature_fod = self.__extract_output(resnet_fod, 'resnet_fod', input_fod)
   
        vgg16_fod = tf.keras.applications.VGG16(weights=self.fine_tune_model_file, include_top=False)
        feature_fod = self.__extract_output(vgg16_fod, 'vgg16_fod', input_fod)
        
        feature_fod = self.__dense(feature_fod)
        output_fod = tf.keras.layers.Dense(name='fod_output', units=self.class_num, activation=tf.nn.sigmoid)(feature_fod)
        
        model = tf.keras.Model(inputs=[input_fod], outputs=[output_fod])

        losses = {
            'fod_output': 'binary_crossentropy',
        }
        
        model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=losses,
                      metrics=['accuracy'])
        return model
    
    def fit(self, model_file, checkpoint_dir, log_dir, max_epoches=EPOCHS, train=True):
        self.model.summary()

        if not train:
            self.model.load_weights(model_file)
        else:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir,
                                                             save_weights_only=True,
                                                             save_best_only=True,
                                                             period=2,
                                                             verbose=1)
            earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            mode='min',
                                                            min_delta=0.001,
                                                            patience=3,
                                                            verbose=1)

            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

            input_name_list = ['fod_input']
            output_name_list = ['fod_output']
            self.model.fit_generator(generator=self.dataset.data_generator(input_name_list, output_name_list, 'train.txt'),
                                     epochs=max_epoches,
                                     steps_per_epoch=self.dataset.train_num() // self.batch_size,
                                     validation_data=self.dataset.data_generator(input_name_list, output_name_list, 'val.txt'),
                                     validation_steps=self.dataset.val_num() // self.batch_size,
                                     callbacks=[cp_callback, earlystop_cb, tb_callback],
                                     max_queue_size=10,
                                     workers=1,
                                     verbose=1)

            self.model.save(model_file)

    def predict(self):
        input_name_list = ['fod_input']
        output_name_list = ['fod_output']
        predictions = self.model.predict_generator(generator=self.dataset.data_generator(input_name_list, output_name_list, 'test.txt', shuffle=False),
                                                   steps=self.dataset.test_num() // self.batch_size,
                                                   verbose=1)
        if len(predictions) > 0:
            fod_preds = predictions
            # print(fod_preds)
            test_data = self.dataset.data_generator(input_name_list, output_name_list, 'test.txt', shuffle=False)
            correct = 0
            steps = self.dataset.test_num() // self.batch_size
            total = steps * self.batch_size
    
            for step in range(steps):
                _, test_batch_y = next(test_data)
                fod_real_batch = test_batch_y['fod_output']
                for i, fod_real in enumerate(fod_real_batch):
                    fod_real = fod_real.tolist()
                    one_num = fod_real.count(1)
                    fod_pred_idxs = sorted(list(map(fod_preds[self.batch_size * step + i].tolist().index,
                                             heapq.nlargest(one_num, fod_preds[self.batch_size * step + i]))))
                    fod_real_idxs = [i for i,x in enumerate(fod_real) if x == 1]
                    # print(fod_pred_idxs)
                    # print(fod_real_idxs)
                    if fod_real_idxs == fod_pred_idxs:
                        correct += 1
    
            print("fod==> correct:{}, total:{}, correct_rate:{}".format(correct, total, 1.0 * correct / total))
        return predictions

    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs[0]['fod_input'], dtype=np.float32)
        batch_x = np.expand_dims(batch_x, 0)
    
        predictions = self.model.predict({'fod_input': batch_x}, batch_size=1)
        # predictions = np.asarray(predictions)
        return predictions
