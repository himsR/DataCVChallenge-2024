from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import os
#from skimage import io, color
from PIL import Image
import pickle

class ImageDataset(Dataset):
	def __init__(self, root_dir, meta_file, transform=None):

		print("building dataset from {}".format(meta_file))

		self.root_dir = root_dir
		self.transform = transform

		info = pd.read_csv(meta_file)
		self.paths = info['path'].values.tolist()
		self.labels = info['label'].values.tolist()
		assert len(self.paths) == len(self.labels)

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):

		path = os.path.join(self.root_dir, self.paths[idx])
		label = self.labels[idx]

#		img = io.imread(path)
#		img = color.gray2rgb(img)
		img = Image.open(path)

		if self.transform:
			img = self.transform(img)

		return img, label

def get_transform(args, is_train=True):
	if is_train:
		_transform = [transforms.RandomCrop(args.input_size, args.input_size),
						transforms.RandomHorizontalFlip(),
						transforms.ColorJitter(0.1, 0.1, 0.1),
						transforms.RandomRotation(degrees=7), 
						transforms.ToTensor()]
	else:
		_transform = [transforms.CenterCrop(args.input_size),
						transforms.ToTensor()]
	if args.normalize:
		_transform.append(transforms.Normalize(
								mean=[0.485, 0.456, 0.406],
								std=[0.229, 0.224, 0.225]))

	return transforms.Compose(_transform)


class FeatureDataset(Dataset):
	def __init__(self, file_path):
		with open(file_path, 'rb') as f:
			self.data = pickle.load(f)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]
