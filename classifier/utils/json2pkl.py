import json
import pdb
import os
import pickle
import numpy as np

data = json.load(open('/home/ubuntu/comp/data/extracted_features/26k.json'))


img_path_list = [x['file_name'].split('pool/')[-1] for x in data['images']]
print(len(img_path_list))

prefix = '/home/ubuntu/comp/data/extracted_features'
with open(os.path.join(prefix, '176k_filename.pkl'), 'rb') as f:
	names = pickle.load(f)

features = np.load(os.path.join(prefix, '176k_feature.npy'))

feature_dict = {}
for i, name in enumerate(names):
	feature_dict[name.split('pool/')[-1]] = features[i]

import pdb
pdb.set_trace()
candidates = []
for img in img_path_list:
	candidates.append(
		[img, feature_dict[img], 0]
	)
		
print(len(candidates))
with open(os.path.join(prefix, '26k', 'all.pkl'), 'wb') as f:
	pickle.dump(candidates, f)