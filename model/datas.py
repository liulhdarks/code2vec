from tensorflow.python.lib.io.file_io import FileIO
import random

random.seed(123)


class CodePath(object):
    def __init__(self):
        self.id = None
        self.label = None
        self.items = []
        self.origin = None
        self.vars = {}


class LabelIndexReader(object):
    def __init__(self):
        self.i2t = {}
        self.t2i = {}

    def record(self, label):
        if label not in self.t2i:
            index = len(self.t2i)
            self.t2i[label] = index
            self.i2t[index] = label

    def get_index(self, label):
        return self.t2i[label]

    def get_label(self, index):
        return self.i2t[index]

    def count(self):
        return len(self.i2t)


class TermIndexReader(object):
    def __init__(self, filename):
        self.i2t = {}
        with FileIO(filename, mode="r") as fio:
            lines = fio.readlines()
            for line in lines:
                line = line.strip(' \r\n\t')
                datas = line.split('\t')
                self.i2t[int(datas[0])] = datas[1]
        print('load idx ', len(self.i2t))

    def count(self):
        return len(self.i2t)


class DataReader(object):
    def __init__(self, corpus_path, path_idx_path, terminal_idx_path):
        self.path_idxs = TermIndexReader(path_idx_path)
        self.terminal_idxs = TermIndexReader(terminal_idx_path)
        self.label_idx = LabelIndexReader()
        self.datas = []
        self.read_corpus(corpus_path)

    def read_corpus(self, corpus_path):
        with FileIO(corpus_path, mode="r") as fio:
            lines = fio.readlines()
            code_path = None
            flag = 0
            for line in lines:
                line = line.strip(' \r\n\t')
                if line == '':
                    if code_path is not None:
                        random.shuffle(code_path.items)
                        self.datas.append(code_path)
                        code_path = None
                    continue
                if code_path is None:
                    code_path = CodePath()
                    if line.startswith('#'):
                        code_path.id = int(line[1:])
                else:
                    if line.startswith('label:'):
                        code_path.label = line[len('label:'):]
                        self.label_idx.record(code_path.label)
                    elif line.startswith('class:'):
                        code_path.origin = line[len('class:'):]
                    elif line.startswith('paths:'):
                        flag = 1
                    elif line.startswith('vars:'):
                        flag = 2
                    else:
                        if flag == 1:
                            datas = line.split('\t')
                            code_path.items.append((int(datas[0]), int(datas[1]), int(datas[2])))
                        elif flag == 2:
                            datas = line.split('\t')
                            code_path.vars[datas[1]] = datas[0]

            if code_path is not None:
                random.shuffle(code_path.items)
                self.datas.append(code_path)
        random.shuffle(self.datas)
        print('load corpus ', len(self.datas), ' labels:', self.label_idx.count())


if __name__ == '__main__':
    corpus_path = '/Users/lihua.llh/Documents/codes/java/workdir/code/model/corpus.txt'
    path_idx_path = '/Users/lihua.llh/Documents/codes/java/workdir/code/model/path_idxs.txt'
    terminal_idx_path = '/Users/lihua.llh/Documents/codes/java/workdir/code/model/terminal_idxs.txt'
    reader = DataReader(corpus_path, path_idx_path, terminal_idx_path)




