from model.datas import *
import tensorflow as tf
import numpy as np


class DatasetBuilder(object):
    def __init__(self, reader, opt):
        test_count = int(len(reader.datas) * 0.2)
        test_datas = reader.datas[0: test_count]
        train_datas = reader.datas[test_count:]
        print('train count:', len(train_datas), ' test count:', len(test_datas))
        train_datas = self.build_datas(reader, train_datas, opt)
        test_datas = self.build_datas(reader, test_datas, opt)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_datas)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(test_datas)
        self.train_dataset = self.train_dataset.batch(32)
        self.train_dataset = self.train_dataset.shuffle(buffer_size=160)
        self.test_dataset = self.test_dataset.batch(32)
        self.test_dataset = self.test_dataset.shuffle(buffer_size=160)

    def build_datas(self, reader, datas, opt):
        inputs_start = []
        inputs_path = []
        inputs_end = []
        labels = []
        for code_path in datas:
            label_idx = reader.label_idx.get_index(code_path.label)
            labels.append(label_idx)
            start_list = []
            path_list = []
            end_list = []
            for start, path, end in code_path.items:
                start_list.append(start)
                path_list.append(path)
                end_list.append(end)
            start_list = self.pad_inputs(start_list, opt.max_path_length)
            path_list = self.pad_inputs(path_list, opt.max_path_length)
            end_list = self.pad_inputs(end_list, opt.max_path_length)
            inputs_start.append(start_list)
            inputs_path.append(path_list)
            inputs_end.append(end_list)
        print('inputs_start count:', len(inputs_start))
        result = {}
        result['inputs_start'] = np.array(inputs_start, dtype=int)
        result['inputs_path'] = np.array(inputs_path, dtype=int)
        result['inputs_end'] = np.array(inputs_end, dtype=int)
        result['labels'] = np.array(labels, dtype=int)
        return result

    def pad_inputs(self, inputs, pad_length, pad_value=0):
        while len(inputs) < pad_length:
            inputs.append(pad_value)
        return inputs