import numpy as np
import gzip
import os
import platform
import pickle
from keras.utils import to_categorical

"""数据类：包含训练和测试的数据及标签"""
class GetDataSet(object):
    # dataSetName:数据集名称 isIID:是否独立同分布
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0  #

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        else:
            pass

    def mnistDataSetConstruct(self, isIID):
        data_dir = r'.\data\MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]  # 第一维是否相等
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)  # 随机打乱
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')  # 32位大端读取
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]  # 将数据流转化为ndarry对象


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # magic number即魔术码，用于校验文件格式
        magic = _read32(bytestream)  # 读取二进制文件前4字节得到魔术码
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)  # 读取5-8字节，得到图像数量60000
        rows = _read32(bytestream)  # 读取第9-12字节，得到图像行数/高度，为28
        cols = _read32(bytestream)  # 读取第13-16字节，得到图像列数/宽度，为28
        buf = bytestream.read(rows * cols * num_images)  # 读取像素值
        data = np.frombuffer(buf, dtype=np.uint8)  # 将数据流转化为ndarry字节对象
        data = data.reshape(num_images, rows, cols, 1)  # 数组重构。最后面的'1'表示什么意思？
        return data


def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    """one-hot: 只有一个值为1的向量。标签为4转化为one-hot后为：00010000000 """
    return to_categorical(labels_dense)

def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)  # 读取标签信息，即为0-9中的一个
        return dense_to_one_hot(labels)


if __name__ == "__main__":
    'test data set'
    mnistDataSet = GetDataSet('mnist', True)  # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
