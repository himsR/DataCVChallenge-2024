import os
import logging
import shutil
import numpy as np
import pdb

import torch
import torch.nn as nn

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, length=0):
		self.length = length
		self.reset()

	def reset(self):
		if self.length > 0:
			self.history = []
		else:
			self.count = 0
			self.sum = 0.0
		self.val = 0.0
		self.avg = 0.0

	def update(self, val):
		if self.length > 0:
			self.history.append(val)
			if len(self.history) > self.length:
				del self.history[0]
			self.val = self.history[-1]
			self.avg = np.mean(self.history)
		else:
			self.val = val
			self.sum += val
			self.count += 1
			self.avg = self.sum / self.count

def learning_rate_decay(optimizer, t, lr_0):
	for param_group in optimizer.param_groups:
		lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
		param_group['lr'] = lr

def create_logger(name, log_file, rank=0, level=logging.INFO):
	l = logging.getLogger(name)
	formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s][rank:{}] %(message)s'.format(rank))
	fh = logging.FileHandler(log_file)
	fh.setFormatter(formatter)
	sh = logging.StreamHandler()
	sh.setFormatter(formatter)
	l.setLevel(level)
	l.addHandler(fh)
	l.addHandler(sh)
	return l

def load_checkpoint(model, optimizer, ckpt_path, data_parallel=False):

	checkpoint = torch.load(ckpt_path)

	ms = model.module if data_parallel else model
	ms.load_state_dict(checkpoint['model'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	best_acc = checkpoint['best_acc']
	start_epoch = checkpoint['epoch']

	return start_epoch, best_acc


def save_checkpoint(state, is_best_acc, data_parallel, filename):
	ms = state['model']
	ms = ms.module if data_parallel else ms
	state['model'] = ms.state_dict()

	torch.save(state, filename+'.pyth')
	if is_best_acc:
		shutil.copyfile(filename+'.pyth', filename+'_best_acc.pyth')
	
def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	import pdb
	pdb.set_trace()

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def accuracy_binary(output, target):
	""" compute binary accuracy """
	pred = (output > 0).view(-1)
	correct = pred.eq(target.byte())
	# import pdb
	# pdb.set_trace()
	return correct.float().mean()


