import multiprocessing
import os.path as op
from threading import local
from zipfile import ZipFile, BadZipFile

from PIL import Image
from io import BytesIO
import torch.utils.data
from torch import randperm
import numpy as np

_VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.bmp', '.png']

class ZippedDataset(torch.utils.data.Dataset):
    _IGNORE_ATTRS = {'_zip_file'}

    def __init__(self, path,
                 transform=None,
                 extensions=None):
        self._path = path
        if not extensions:
            extensions = _VALID_IMAGE_TYPES
        self._zip_file = ZipFile(path)
        self.zip_dict = {}
        self.samples = []
        self.transform = transform
        self.class_count = [0,0]

        for fst in self._zip_file.infolist():
            fname = fst.filename
            if 'no_ped' in fname:
                target = 0
                self.class_count[0] += 1
            else:
                target = 1
                self.class_count[1] += 1

            if target is None:
                continue
            if fname.endswith('/') or fname.startswith('.') or fst.file_size == 0:
                continue
            ext = op.splitext(fname)[1].lower()
            if ext in extensions:
                self.samples.append((fname, target))
        assert len(self), "No images found in: {} ".format(self._path)

    def __repr__(self):
        return 'ZipData({}, size={})'.format(self._path, len(self))

    def __getstate__(self):
        return {
            key: val if key not in self._IGNORE_ATTRS else None
            for key, val in self.__dict__.items()
        }

    def __getitem__(self, index):
        proc = multiprocessing.current_process()
        pid = proc.pid # get pid of this process.
        if pid not in self.zip_dict:
            self.zip_dict[pid] = ZipFile(self._path)
        zip_file = self.zip_dict[pid]

        if index >= len(self) or index < 0:
            raise KeyError("{} is invalid".format(index))
        path, target = self.samples[index]
        try:
            sample = Image.open(BytesIO(zip_file.read(path))).convert('RGB')
        except BadZipFile:
            print("bad zip file")
            return None, None
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class transformWrapper(torch.utils.data.Dataset):
    """
    Applies a transform to a dataset
        :param torch.utils.data.Dataset:
        :param pytroch transform: 
    """
    def __init__(self, dataset, transform):
        super(transformWrapper, self).__init__()
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, idx):
        return self.transform(self.dataset[idx][0]), self.dataset[idx][1]

    def __len__(self):
        return len(self.dataset)


def splitTrainTest(dataset, lengths, transform_train, transform_test, mode='deterministic'):
    N = len(dataset)
    assert sum(lengths) == N
    assert len(lengths) == 2
    if mode == 'deterministic':
        print('Deterministic Split')
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        indices = randperm(sum(lengths)).tolist()
    else:
        print('Random Split')
        indices = randperm(sum(lengths)).tolist()
    train_data, test_data = [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(np.cumsum(lengths), lengths)]

    train_data = transformWrapper(train_data, transform_train)
    test_data = transformWrapper(test_data, transform_test)
    return train_data, test_data


if __name__ == '__main__':
    from IPython import embed
    import os
    from torchvision import transforms, datasets

    # valpath = os.path.join(os.path.expanduser('~/Desktop'), 'val.zip')
    # val_map = os.path.join(os.path.expanduser('~/Desktop'), 'val_map.txt')
    # dataset = ZipData(valpath, val_map,
    #                         transforms.Compose([
    #                         transforms.Resize(256),
    #                         transforms.CenterCrop(224),
    #                         transforms.ToTensor(),
    #                         ]))
    embed()
    path = os.path.join(os.path.expanduser('~/Desktop/datasets/pedestrian_recognition'), 'dataset.zip')
    dataset = ZippedDataset(path, 
                            transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            ]))

