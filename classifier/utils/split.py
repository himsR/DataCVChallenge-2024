import numpy as np
import pickle
import os
import pdb
import random


prefix = '/home/ubuntu/comp/data/extracted_features'
with open(os.path.join(prefix, '176k_filename.pkl'), 'rb') as f:
	names = pickle.load(f)

with open(os.path.join(prefix, 'train_filename.pkl'), 'rb') as f:
	train_names = pickle.load(f)

features = np.load(os.path.join(prefix, '176k_feature.npy'))
train_features = np.load(os.path.join(prefix, 'train.npy'))

train_list = []
val_list = []
for i, name in enumerate(names):
	if 'cityscapes_train' in name or 'kitti_train' in name:
		if random.random() > 0.1:
			train_list.append(
				[name, features[i], 0]
				)
		else:
			val_list.append(
				[name, features[i], 0]
				)
	if 'detrac' in name:
		if random.random() < 0.09:
			train_list.append(
				[name, features[i], 1]
				)
		elif random.random() < 0.009:
			val_list.append(
				[name, features[i], 1]
				)
# train_list.extend(train_list)
import pdb
pdb.set_trace()
print(len(val_list))
for i, name in enumerate(train_names):
	if int(name.split('/')[-2]) > 90 and random.random() <= 0.567:
		val_list.append(
			[name, train_features[i], 1]
		)
	else:
		train_list.append(
			[name, train_features[i], 1]
			)
print(len(val_list))

with open(os.path.join(prefix, 'train_list.pkl'), 'wb') as f:
	pickle.dump(train_list, f)

with open(os.path.join(prefix, 'val_list.pkl'), 'wb') as f:
	pickle.dump(val_list, f)

