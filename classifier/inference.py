from torch.utils.data import DataLoader
import torch
from utils.datasets import FeatureDataset
import argparse
from models.classifier import fc_Classifier
from utils.util import load_checkpoint
import yaml

parser = argparse.ArgumentParser(description='image-level-classifier')
parser.add_argument('--val-path', default='cfgs/config.yaml', help='set config')
parser.add_argument('--resume', default=None, type=str, help='resume from a checkpoint')

parser.add_argument('--cfg', default='cfgs/inf.yaml', help='set config')
parser.add_argument('--inf-output', default='logs/inference_score.pkl', help='set config')
args = parser.parse_args()

with open(args.cfg) as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
for k, v in config.items():
	setattr(args, k, v)


val_set = FeatureDataset(
        args.val_path
    )

val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=1024,
                shuffle=False, 
                pin_memory=True, drop_last=False)

model = fc_Classifier()
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
_, best_acc = load_checkpoint(model, optimizer, args.resume)

all_pred = []
all_names = []
for i, (name, inputs, _) in enumerate(val_loader):
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        # labels = labels.cuda().float()
    
    pred = model(inputs)
    all_pred.extend(pred.cpu().detach().numpy())
    all_names.extend(name)

data = {}
for i, name in enumerate(all_names):
    data[name] = all_pred[i]
import pickle
with open(args.inf_output, 'wb') as f:
    pickle.dump(data, f)




                        
    