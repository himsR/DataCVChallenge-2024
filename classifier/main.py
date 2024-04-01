from models.classifier import fc_Classifier
import numpy as np
import random
import argparse
import os
import time
import pdb
import logging
import yaml

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils.util import create_logger, AverageMeter, accuracy_binary, save_checkpoint, load_checkpoint
import utils.distributed as du
from utils.datasets import FeatureDataset, get_transform

# argparser
parser = argparse.ArgumentParser(description='image-level-classifier')
parser.add_argument('--resume', default=None, type=str, help='resume from a checkpoint')
parser.add_argument('--cfg', default='cfgs/config.yaml', help='set config')
parser.add_argument('--num_gpus', default=1, type=int)
args = parser.parse_args()

with open(args.cfg) as f:
	config = yaml.load(f, Loader=yaml.FullLoader)
for k, v in config.items():
	setattr(args, k, v)

best_acc = 0
start_epoch = 0

def main():
	global args, best_acc, start_epoch
	
	os.makedirs('{}'.format(args.save_path), exist_ok=True)
	os.makedirs('{}/ckpt'.format(args.save_path), exist_ok=True)

	# logging configuration
	logger = create_logger('global_logger', log_file=os.path.join(args.save_path,'log.txt'))
	logger.info('{}'.format(args))
	tb_logger = SummaryWriter(args.save_path)

	# Construct model
	model = fc_Classifier()
	
	# Send to cuda and set up for multi-gpu
	cur_device = torch.cuda.current_device()
	model = model.cuda(device=cur_device)
	if args.num_gpus > 1:
		model = torch.nn.parallel.DistributedDataParallel(
				modeule=model, device_ids=[cur_device], output_device=cur_device)
	if du.is_master_proc():
		print("=> created model '{}'".format(args.arch))
	

	# optimizer
	optimizer = torch.optim.Adam(model.parameters(), 
										lr=args.base_lr,
										betas=(0.9, 0.999),
										weight_decay=1e-6)

	# loss function
	loss_func = nn.BCEWithLogitsLoss(reduction="mean")
	if torch.cuda.is_available():
		loss_func = loss_func.cuda()

	# optionally resume from a checkpoint
	if args.auto_resume:
		ckpt_path = os.path.join(args.save_path, 'ckpt', 'ckpt.pyth')
		if os.path.exists(ckpt_path):
			logger.info("=> loading checkpoint '{}'".format(ckpt_path))
			start_epoch, best_acc = load_checkpoint(model, optimizer, ckpt_path)
	elif args.resume:
		logger.info("=> loading checkpoint '{}'".format(args.resume))
		start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume)
	else:
		logger.info("=> start training from scratch")


	# construct dataloader
	train_set = FeatureDataset(
		args.train_path)
	val_set = FeatureDataset(
		args.val_path
	)

	train_sampler = DistributedSampler(train_set) if args.num_gpus > 1 else None
	val_sampler = DistributedSampler(val_set) if args.num_gpus > 1 else None
	train_loader = torch.utils.data.DataLoader(
					train_set, batch_size=args.batch_size,
					shuffle=(False if train_sampler else True),
					sampler=train_sampler, num_workers=args.num_workers, 
					pin_memory=True, drop_last=True)
	val_loader = torch.utils.data.DataLoader(
					val_set, batch_size=args.batch_size,
					sampler=val_sampler, shuffle=False, 
					pin_memory=True, drop_last=False)
	
	for epoch in range(start_epoch, args.epochs):
		
		# training
		train_epoch(train_loader, model, loss_func, optimizer, epoch, tb_logger)

		# evaluation 
		if (epoch+1) % args.eval_freq == 0 or epoch + 1 == args.epochs:

			loss, acc = eval_epoch(val_loader, model, loss_func, epoch, tb_logger)
	
			# saving checkpoint
			is_best_acc = acc > best_acc
			best_acc = max(acc, best_acc)
			save_checkpoint({
				  'epoch': epoch, 
				  'model': model, 
				  'best_acc': best_acc,
				  'optimizer': optimizer.state_dict()}, 
				  is_best_acc, (args.num_gpus > 1),
				  os.path.join(args.save_path, 'ckpt', 'ckpt')) 


def train_epoch(train_loader, model, loss_func, optimizer, epoch, tb_logger):
		
	freq = args.log_freq

	batch_time = AverageMeter(freq)
	data_time = AverageMeter(freq)
	losses = AverageMeter(freq)
	accs = AverageMeter(freq)

	logger = logging.getLogger('global_logger')

	model.train()

	end = time.time()

	for i, (filenames, inputs, labels) in enumerate(train_loader):
		data_time.update(time.time() - end)
		if torch.cuda.is_available():
			inputs = inputs.cuda()
			labels = labels.cuda().float()
		
		pred = model(inputs)
		batch_size = inputs.size(0)
		loss = loss_func(pred.view(batch_size), labels)

		optimizer.zero_grad()
		loss.backward()

		optimizer.step()

		# cuda synchronize
		#if torch.cuda.is_availabel():
		#	torch.cuda.synchronize()

		# TODO: other metrics
	
		acc = accuracy_binary(pred, labels)

		if args.num_gpus > 1:
			loss, acc = du.all_reduce([loss, acc])
		
		losses.update(loss.item())
		accs.update(acc.item())

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if du.is_master_proc() and (i+1) % args.log_freq == 0:	
			step = epoch * len(train_loader) + i
			tb_logger.add_scalar('loss', losses.avg, step)
			tb_logger.add_scalar('acc', accs.avg, step)
			logger.info('Train Epoch: [{0}/{1}][{2}/{3}]\t'
				  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss: {loss.avg:.3f}\t'
				  'Accuracy: {accs.avg: .3f}\t'.format(
					epoch, args.epochs, i, len(train_loader), 
					batch_time=batch_time,
					data_time=data_time, loss=losses, accs=accs))


def eval_epoch(val_loader, model, loss_func, epoch, tb_logger):

	logger = logging.getLogger('global_logger')
	losses = AverageMeter()
	accs = AverageMeter()
	
	model.eval()

	for i, (file_names, inputs, labels) in enumerate(val_loader):
		if torch.cuda.is_available():
			inputs = inputs.cuda()
			labels = labels.cuda().float()

		pred = model(inputs)

		batch_size = inputs.size(0)
		loss = loss_func(pred.view(batch_size), labels)
		acc = accuracy_binary(pred, labels)
	
		if args.num_gpus > 1:
			loss, acc = du.all_reduce([loss, acc])
		losses.update(loss.item())
		accs.update(acc.item())

	logger.info('Eval Epoch: [{}/{}]\t'
				'eval loss: {loss.avg:.3f}\t'
				'eval acc: {acc.avg:.3f}\t'.format(
					epoch, args.epochs, loss=losses, acc=accs))
	tb_logger.add_scalar('val/loss', losses.avg, epoch)
	tb_logger.add_scalar('val/acc', accs.avg, epoch)

	return losses.avg, accs.avg


if __name__ == '__main__':
	main()
