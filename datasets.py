import os
import numpy as np
import cv2
import random
import tensorflow as tf


class DataSet:
    def __init__(self, proj_dir, data_dir, batch_size=64, input_size=64, fine_tune=False):
        self.proj_dir = proj_dir
        self.data_dir = os.path.join(proj_dir, data_dir)
        self.batch_size = batch_size
        self.input_size = input_size
        self.fine_tune = fine_tune
        self.__train_num, self.__val_num, self.__test_num = self.__get_samples_num(os.path.join(self.data_dir, 'train.txt'),
                                                               os.path.join(self.data_dir, 'val.txt'),
                                                               os.path.join(self.data_dir, 'test.txt'))

    def __get_samples_num(self, train_label_file, val_label_file, test_label_file):
        train_num = 0
        val_num = 0
        test_num = 0
        if not os.path.exists(train_label_file) or \
                not os.path.exists(val_label_file) or \
                not os.path.exists(test_label_file):
            return train_num, val_num, test_num
        
        with open(train_label_file) as f:
            train_num = len(f.readlines())
        with open(val_label_file) as f:
            val_num = len(f.readlines())
        with open(test_label_file) as f:
            test_num = len(f.readlines())
        return train_num, val_num, test_num


    def __load_input_img(self, proj_dir, file_name, fine_tune=False):
        img_path = os.path.join(proj_dir, file_name)
        
        # print(img_path)
        if fine_tune:
            img = cv2.imread(img_path)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # print(img)
        img = cv2.resize(img, (self.input_size, self.input_size)) / 255
        
        return img

    def load_input_imgpath_label(self, file_name, labels_num=1, shuffle=True):
        imgpath = []
        labels = []

        with open(os.path.join(self.data_dir, file_name)) as f:
            lines_list = f.readlines()
            if shuffle:
                random.shuffle(lines_list)
           
            for lines in lines_list:
                line = lines.rstrip().split(',')
                label = []
                if labels_num == 1:
                    label = int(line[1])
                else:
                    lab = line[1].split(' ')
                    for i in range(labels_num):
                        label.append(int(lab[i]))
                imgpath.append(line[0])
                labels.append(label)
        return np.array(imgpath), np.array(labels)

    def train_num(self):
        return self.__train_num

    def val_num(self):
        return self.__val_num

    def test_num(self):
        return self.__test_num

    def load_batch_data_label(self, filename_list, label_list, label_num=1, shuffle=True):
        file_num = len(filename_list)
        if shuffle:
            idx = np.random.permutation(range(file_num))
            filename_list = filename_list[idx]
            label_list = label_list[idx]
        max_num = file_num - (file_num % self.batch_size)
        for i in range(0, max_num, self.batch_size):
            batch_x = []
            batch_y = []
            for j in range(self.batch_size):
                img = self.__load_input_img(self.proj_dir, filename_list[i + j], self.fine_tune)
                if not self.fine_tune:
                    img = np.resize(img, (self.input_size, self.input_size, 1))
                label = label_list[i + j]
                batch_x.append(img)
                batch_y.append(label)
            batch_x = np.array(batch_x, dtype=np.float32)
            if label_num == 1:
                batch_y = tf.keras.utils.to_categorical(batch_y, 7)
            else:
                batch_y = np.array(batch_y)
            if shuffle:
                idx = np.random.permutation(range(self.batch_size))
                batch_x = batch_x[idx]
                batch_y = batch_y[idx]
            yield batch_x, batch_y


class Cofw(DataSet):
    def __init__(self, proj_dir, data_dir, batch_size=64, input_size=64, class_num=2, fine_tune=False):
        DataSet.__init__(self, proj_dir, data_dir, batch_size, input_size, fine_tune)
        self.class_num = class_num

        print("fod train_num:%d" % self.train_num())
        print("fod val_num:%d" % self.val_num())
        print("fod test_num:%d" % self.test_num())

    def data_generator(self, input_name_list, output_name_list, label_file_name='train.txt', shuffle=True):
        fod_filenames, fod_labels = self.load_input_imgpath_label(label_file_name, labels_num=self.class_num, shuffle=shuffle)
        while True:
            fod_generator = self.load_batch_data_label(fod_filenames, fod_labels, label_num=self.class_num, shuffle=shuffle)
            fod_batch_x, fod_batch_y = next(fod_generator)
 
            yield ({input_name_list[0]: fod_batch_x},
                   {output_name_list[0]: fod_batch_y})
