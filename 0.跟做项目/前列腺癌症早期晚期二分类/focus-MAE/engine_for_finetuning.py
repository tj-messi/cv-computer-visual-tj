# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import pathlib
import os
import json
import sys
from multiprocessing import Pool
from typing import Iterable, Optional
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import cv2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import torch
from scipy.special import softmax
from timm.data import Mixup
from timm.utils import ModelEma, accuracy
from einops import rearrange
import utils
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import pandas as pd
# from torchgs import GridSearch
from sklearn.metrics import roc_auc_score
from scipy import ndimage
# import imageio

import torch.nn as nn

def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2*loss


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]

    loss = torch.sum((arr2-arr)**2)

    return lamda1*loss

import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = -(targets * logits.log() + (1 - targets) * (1 - logits).log())
        return ce_loss * ((1 - p) ** self.gamma) * self.alpha


def train_class_batch(model, samples, target, criterion):

    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs

def train_class_batch_vivit(model, samples, target, criterion):
    outputs = model(samples)
    loss= ((outputs.logits - target)**2).mean()
    return loss, outputs.logits

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(
        optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    log_writer=None,
                    start_steps=None,
                    model_type = "vit",
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None,
                    run= None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    
    for data_iter_step, (samples, targets, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it%len(lr_schedule_values)] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        

        # else:
        targets = targets.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)

        if mixup_fn is not None:
            # mixup handle 3th & 4th dimension
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
    
            if model_type=='vivit':
                samples = samples.view(B, T,C, H, W)
            else:
                samples = samples.view(B,C,T,H,W)

        if loss_scaler is None:
            # samples = samples.half()
            with torch.cuda.amp.autocast(dtype=torch.float32):
                if model_type=="vivit":
                    loss, output = train_class_batch_vivit(model, samples, targets,
                                             criterion)
                else:
                    
                    outputs = model(samples)
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, targets)
        else:
            with torch.cuda.amp.autocast(dtype=torch.float32):
                if model_type=="vivit":
                    loss, output = train_class_batch_vivit(model, samples, targets,
                                             criterion)
                else:
                    outputs = model(samples)
                    outputs = F.softmax(outputs, dim=1)
                    loss = criterion(outputs, targets)
                

        loss_value = loss.item()
        
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if run:
            run['train/loss'].append(loss_value)
        if loss_scaler is None:
            loss /= update_freq
            
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            
            model.step()
            
            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            
            model.step()
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

       
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_masking_recons(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    save_path : str,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    model_type="vit",
                    log_writer=None,
                    start_steps=None,
                    lr_schedule_values=None,
                    wd_schedule_values=None,
                    num_training_steps_per_epoch=None,
                    update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter(
        'min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    if save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)


    for data_iter_step, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        
        step = data_iter_step // update_freq
        samples = batch[0]
        

        targets = batch[1]
        ids = batch[2]
        mask = batch[3]
        
        # print("print batch ",batch)
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it%len(lr_schedule_values)] * param_group[
                        "lr_scale"]
                if wd_schedule_values is not None and param_group[
                        "weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).flatten(1).to(torch.bool)
        
        if mixup_fn is not None:
            # mixup handle 3th & 4th dimension
            B, C, T, H, W = samples.shape
            samples = samples.view(B, C * T, H, W)
            samples, targets = mixup_fn(samples, targets)
            samples = samples.view(B, C, T, H, W)

        if loss_scaler is None:
            # samples = samples.half()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(samples, mask)
                
                loss = criterion(output, targets)
                

        else:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(samples,  mask)
                loss = criterion(output, targets)
               

        
        loss_value = loss.item()
        samples = samples.cpu().numpy()
        targets = targets.cpu().numpy()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            
            model.backward(loss)
            grad_norm = model.get_global_grad_norm()
            
            model.step()
            
            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(
                optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        
        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device,args, model_type="vit"):
    # criterion = torch.nn.BCELoss(size_average=True)
    # criterion = SoftTargetCrossEntropy()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss(size_average=True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    all_labels = []
    all_preds = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        FNAME = batch[2]

        print(args.masking)
        if args.masking:

            mask = batch[3]
            mask = mask[0].to(device, non_blocking=True).flatten(1).to(torch.bool)
        # print(mask)
        images = images.to(device, non_blocking=True)
        if model_type=="vivit":
            images = images.transpose(1,2)
        target = target.to(device, non_blocking=True)

        target_new = torch.zeros(target.shape[0],2).cuda()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            if args.masking:
                output= model(images, mask)
            else:
                output = model(images)
            if model_type=="vivit":
                output = output.logits

            loss = criterion(output ,target )

        acc1, acc2 = accuracy(output, target, topk=(1,2))
        probs = torch.sigmoid(output)
        print(probs)
        print(target)
        all_labels.extend(target.cpu().numpy())  # Add true labels to the list
        all_preds.extend(probs.cpu().numpy())  # Add predicted probabilities to the list

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

    # AUC-score
    all_labels = torch.tensor(all_labels)
    all_preds = torch.tensor(all_preds)

    # auc_score = roc_auc_score(all_labels.numpy(), all_preds.numpy()[:, 1])  # AUC for binary classification

    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc2,
            losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def final_test_2(data_loader, model, device, file,args, model_type="vit", run=None):
    criterion =  SoftTargetCrossEntropy()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    

    for batch in metric_logger.log_every(data_loader, 100, header):
    # for batch in data_loader:
        
        images = batch[0]
    
        target = batch[1]
        ids = batch[2]
   
        chunk_nb = batch[3]
        split_nb = batch[4]
        try:
            indices = batch[5]
            

        except:
            pass
        
        if args.masking:

            mask = batch[5][0].to(device, non_blocking=True).flatten(1).to(torch.bool)
        images = images.to(device, non_blocking=True)
        if model_type=="vivit":
            images = images.transpose(1,2)
            
        target = target.to(device, non_blocking=True)

        target_new = torch.zeros(target.shape[0], 2).to(device, non_blocking=True)
        target_new[:,1] = target
        target_new[:,0] = 1- target

        
        # compute output
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if args.masking:
                output= model(images,mask)
            else:
                output = model(images)
            
            
            loss = criterion(output, target_new)
        try:
            for i in range(output.size(0)):
                string = "{} {} {} {} {} {}\n".format(
                    ids[i], str(output.data[i].cpu().numpy().tolist()),
                    str(int(target[i].cpu().numpy())),
                    str(int(chunk_nb[i].cpu().numpy())),
                    str(int(split_nb[i].cpu().numpy())),
                    indices[i].cpu().numpy()
                    )

                final_result.append(string)
        except:

            for i in range(output.size(0)):
                string = "{} {} {} {} {}\n".format(
                    ids[i], str(output.data[i].cpu().numpy().tolist()),
                    str(int(target[i].cpu().numpy())),
                    str(int(chunk_nb[i].cpu().numpy())),
                    str(int(split_nb[i].cpu().numpy()))
                    )

                final_result.append(string)

        acc1,acc5 = accuracy(output, target, topk=(1,5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    
    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1,acc5))
        
        for line in final_result:
            f.write(line)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def final_test(data_loader, model, device,vid_atts,vid_imgs, file,args, model_type ="vit"):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]

        
        target = batch[1]
        ids = batch[2]
        
        if args.masking:
            mask = batch[5][0].to(device, non_blocking=True).flatten(1).to(torch.bool)

        for id in ids:
            if not id in vid_atts.keys():
                vid_atts[id] = []
            if not id in vid_imgs.keys():
                vid_imgs[id] = []

        chunk_nb = batch[3]
        split_nb = batch[4]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)



        target_new = torch.zeros(target.shape[0], 2).to(device, non_blocking=True)
        target_new[:,1] = target
        target_new[:,0] = 1- target

        
        # compute output
        with torch.cuda.amp.autocast(dtype=torch.float16):
            if args.masking:
                output= model(images,mask)
            # else:
            #     output = model(images)
            elif model_type=="vivit":
                output = model(images)
                output = output.logits
            elif model_type == "weakly_polyp":
                output, _ = model(images)
            else:
                output = model(images)
           
            loss = criterion(output.to(dtype=float), target_new)

      
        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())),
                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1,acc5 = accuracy(output, target, topk=(1,5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1,acc5))
        
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, vid_atts, vid_imgs



@torch.no_grad()
def final_inference(data_loader, model, device, file,args, model_type="vit"):
  
    criterion = torch.nn.CrossEntropyLoss()
    # switch to evaluation mode
    model.eval()
    final_result = []
    

    for batch in data_loader:
        # print(batch)
        images = batch[0]

        target = batch[1] 
        ids = batch[2]
        

        
        if args.masking:

            mask = batch[3][0].to(device, non_blocking=True).flatten(1).to(torch.bool)
        images = images.to(device, non_blocking=True)
        if model_type=="vivit":
            images = images.transpose(1,2)
            
        target = target.to(device, non_blocking=True)

        target_new = torch.zeros(target.shape[0], 2).to(device, non_blocking=True)
        target_new[:,1] = target
        target_new[:,0] = 1- target

        
        # compute output
        with torch.cuda.amp.autocast():
            if args.masking:
                output= model(images,mask)
            # else:
            #     output = model(images)
            elif model_type=="vivit":
                output = output.logits
            elif model_type == "weakly_polyp":
                output, _ = model(images)
            else:
                output = model(images)
           
            loss = criterion(output, target_new)

        for i in range(output.size(0)):
            string = "{} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()),
                str(int(target[i].cpu().numpy())))
            
            final_result.append(string)

        print(output, target)
        acc1 ,_= accuracy(output, target, topk=(1,2))
        

        print(f"acc1 {acc1}")
        batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}\n".format(acc1))
        
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print(
    #     '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(
    #         top1=metric_logger.acc1,
    #         top5=metric_logger.acc5,
    #         losses=metric_logger.loss))

    return {}

def merge(eval_path, num_tasks, method='prob'):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    # video_pred = open("video_pred", 'a+') #add the predictions for each video to this 

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            
            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            
            if method == 'prob':
                dict_feats[name].append(softmax(data))
                
            else:
                dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    # print("Computing final results")

    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


def merge_vid_class(eval_path, num_tasks,args, method='prob', pred_column="vit_pred", run=None):
    assert method in ['prob', 'score']
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    vid_res = {}
    print("Reading individual output files")
    
    indices = None
    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            # try:
            if indices:
                indices = line.split(']')[1].split(' ')[4]

            data = np.fromstring(
                line.split('[')[1].split(']')[0], dtype=float, sep=',')
            if name not in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            
            if method == 'prob':
                dict_feats[name].append(softmax(data))
                if name in vid_res.keys():
                    vid_res[name].append((np.argmax(softmax(data)), int(label), indices))
                else:
                    vid_res[name] = [(np.argmax(softmax(data)), int(label), indices)]
            else:
                dict_feats[name].append(data)
            dict_label[name] = label
    print("Computing final results")

    
    acc = 0
    
    clip_labels = ""
    label = []
    preds = []
    for keys in vid_res.keys():
        
        max_label = max([x[1] for x in vid_res[keys]])

        if args.test_randomization or  args.sampling_scheme=="scheme1":
            max_pred = max([x[0] for x in vid_res[keys]])
            
        elif args.sampling_scheme=="scheme2":
            max_pred = (sum([x[0] for x in vid_res[keys]])/len(vid_res[keys])) >  0.1
        
        key = keys.strip()
        preds.append(max_pred)
        label.append(max_label)
        print(keys, "label", max_label, "output ", max_pred )
        acc+= (max_label- max_pred)**2

    cm1 = confusion_matrix(label,preds)
    print('Confusion Matrix : \n', cm1)

    
    total1=sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print ('Accuracy : ', accuracy1)
    if run:
        run['video_accuracy'].append(accuracy1)

    sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])

    print('Sensitivity: ', sensitivity)
    if run:
        run['video_sensitivity'].append(sensitivity)

    specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    
    print('specificity : ', specificity )

    if run:
        run['video_specificity'].append(specificity)

    print("MSE LOSS VID Classification ",acc/len(vid_res))
    print("Accuracy  ", )
    input_lst = []
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    p = Pool(64)
    # [pred, top1, top5, label]
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)

    return final_top1 * 100, final_top5 * 100


def merge_ct(eval_path, num_tasks, run=None):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float16, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]


    cm = confusion_matrix(label, pred)
    print("confusion matrix: ", cm)
    specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
    sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])

    print('Specificity  : ', specificity1)
    
    print('Sensitivity : ', sensitivity1 )
    

    accuracy1 = (cm[1,1]+cm[0,0])/sum(sum(cm))
    
    print('Accuracy  : ', accuracy1)

    print(pred, label)
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
