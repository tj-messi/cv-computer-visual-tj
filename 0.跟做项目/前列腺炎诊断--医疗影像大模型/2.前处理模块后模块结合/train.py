#!/usr/bin/env python

import os
import argparse
from datetime import datetime
import builtins
from PIL import Image
import subprocess
import numpy as np
import torch
import torch.distributed as dist

import sys
sys.path.append('./')

import video_dataset
import checkpoint
from VitaCLIP_model import VitaCLIP
from sklearn.metrics import precision_score, recall_score, f1_score

from collections import OrderedDict

def setup_print(is_master: bool):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def main():
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    
    video_dataset.setup_arg_parser(parser)
    checkpoint.setup_arg_parser(parser)

    # train settings
    parser.add_argument('--num_steps', type=int, default=30000,
                        help='number of training steps')
    parser.add_argument('--eval_only', action='store_true',
                        help='run evaluation only')
    parser.add_argument('--save_freq', type=int, default=5000,
                        help='save a checkpoint every N steps')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='evaluate every N steps')
    parser.add_argument('--print_freq', type=int, default=50,
                        help='print log message every N steps')
    parser.add_argument('--lr', type=float, default=4e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='optimizer weight decay')
    parser.add_argument('--batch_split', type=int, default=1,
                        help='optionally split the batch into smaller shards and forward/backward one shard '
                             'at a time to avoid out-of-memory error.')

    # backbone and checkpoint paths
    parser.add_argument('--backbone_path', type=str,
                        help='path to pretrained backbone weights', default='')
    parser.add_argument('--checkpoint_path', type=str,
                        help='path to pretrained checkpoint weights', default=None)
    
    # model params
    parser.add_argument('--patch_size', type=int, default=16,
                        help='patch size of patch embedding')
    parser.add_argument('--num_heads', type=int, default=12,
                        help='number of transformer heads')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='number of transformer layers')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='transformer feature dimension')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='clip projection embedding size')
    parser.add_argument('--mlp_factor', type=float, default=4.0,
                        help='transformer mlp factor')
    parser.add_argument('--cls_dropout', type=float, default=0.5,
                        help='dropout rate applied before the final classification linear projection')

    # zeroshot evaluation
    parser.add_argument('--zeroshot_evaluation', action='store_true', dest='zeroshot_evaluation',
                        help='set into zeroshot evaluation mode')
    parser.add_argument('--zeroshot_text_features_path', type=str, default='./ucf101_text_features_B16/class-only.pth',
                        help='path to saved clip text features to be used for zeroshot evaluation')
    
    #fp16
    parser.add_argument('--use_fp16', action='store_true', dest='fp16',
                        help='disable fp16 during training or inference')
    parser.set_defaults(fp16=False)



    # use summary token attn
    parser.add_argument('--use_summary_token', action='store_true', dest='use_summary_token',
                        help='use summary token')
    # use local prompts
    parser.add_argument('--use_local_prompts', action='store_true', dest='use_local_prompts',
                        help='use local (frame-level conditioned) prompts')
    # use global prompts
    parser.add_argument('--use_global_prompts', action='store_true', dest='use_global_prompts',
                        help='use global (video-level unconditioned) prompts')
    parser.add_argument('--num_global_prompts', type=int, default=8,
                        help='number of global prompts')
    # set defaults
    parser.set_defaults(use_summary_token=False, use_local_prompts=False, use_global_prompts=False)



    # text prompt learning
    parser.add_argument('--use_text_prompt_learning', action='store_true', dest='use_text_prompt_learning',
                        help='use coop text prompt learning')
    parser.add_argument('--text_context_length', type=int, default=77,
                        help='text model context length')
    parser.add_argument('--text_vocab_size', type=int, default=49408,
                        help='text model vocab size')
    parser.add_argument('--text_transformer_width', type=int, default=512,
                        help='text transformer width')
    parser.add_argument('--text_transformer_heads', type=int, default=8,
                        help='text transformer heads')
    parser.add_argument('--text_transformer_layers', type=int, default=12,
                        help='text transformer layers  12origin')
    parser.add_argument('--text_num_prompts', type=int, default=16,
                        help='number of text prompts')
    parser.add_argument('--text_prompt_pos', type=str, default='end',
                        help='postion of text prompt')
    parser.add_argument('--text_prompt_init', type=str, default='',
                        help='initialization to be used for text prompt. Leave empty for random')
    parser.add_argument('--use_text_prompt_CSC', action='store_true', dest='text_prompt_CSC',
                        help='use Class Specific Context in text prompt')
    parser.add_argument('--text_prompt_classes_path', type=str, default='./classes/data_classes.txt',
                        help='path of classnames txt file')


    args = parser.parse_args()

    dist.init_process_group('nccl') 
    setup_print(dist.get_rank() == 0)
    '''cuda_device_id = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(cuda_device_id)'''

    cuda_device_id = dist.get_rank() % (torch.cuda.device_count() - 1) + 2  # 排除0号GPU
    torch.cuda.set_device(cuda_device_id)


    model = VitaCLIP(
        # load weights
        backbone_path=args.backbone_path,
        # data shape
        input_size=(args.spatial_size, args.spatial_size),
        num_frames=args.num_frames,
        # model def
        feature_dim=args.feature_dim,
        patch_size=(args.patch_size, args.patch_size),
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_factor=args.mlp_factor,
        embed_dim=args.embed_dim,
        # use summary token
        use_summary_token=args.use_summary_token,
        # use local prompts
        use_local_prompts=args.use_local_prompts,
        # use global prompts
        use_global_prompts=args.use_global_prompts,
        num_global_prompts=args.num_global_prompts,
        # use text prompt learning
        use_text_prompt_learning=args.use_text_prompt_learning,
        text_context_length=args.text_context_length,
        text_vocab_size=args.text_vocab_size,
        text_transformer_width=args.text_transformer_width,
        text_transformer_heads=args.text_transformer_heads,
        text_transformer_layers=args.text_transformer_layers,
        text_num_prompts=args.text_num_prompts,
        text_prompt_pos=args.text_prompt_pos,
        text_prompt_init=args.text_prompt_init,
        text_prompt_CSC=args.text_prompt_CSC,
        text_prompt_classes_path=args.text_prompt_classes_path,
        # zeroshot eval
        zeroshot_evaluation=args.zeroshot_evaluation,
        zeroshot_text_features_path=args.zeroshot_text_features_path,
    )

    if args.checkpoint_path:
        print('loading checkpoint')
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        renamed_ckpt = OrderedDict((k[len("module."):], v) for k, v in ckpt['model'].items() if k.startswith("module."))
        model.load_state_dict(renamed_ckpt, strict=True)
    
    
    print(model)
    print('----------------------------------------------------')
    print('Trainable Parameters')
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)
    print('----------------------------------------------------')
    model.cuda()
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[cuda_device_id], output_device=cuda_device_id,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    loss_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=args.fp16)
    criterion = torch.nn.CrossEntropyLoss()

    resume_step = checkpoint.resume_from_checkpoint(model, optimizer, lr_sched, loss_scaler, args)

    val_loader = video_dataset.create_val_loader(args)
    if args.eval_only:
        print('Running in eval_only mode.')
        model.eval()
        evaluate(model, val_loader)
        return
    else:
        assert args.train_list_path is not None, 'Train list path must be specified if not in eval_only mode.'
        train_loader = video_dataset.create_train_loader(args, resume_step=resume_step)

    assert len(train_loader) == args.num_steps - resume_step
    batch_st, train_st = datetime.now(), datetime.now()
    misclassfied_1_as_0 = 0
    misclassfied_2_as_0 = 0
    misclassfied_0_as_1or2 = 0
    total_label_1 = 0
    total_label_2 = 0
    total_label_0 = 0
    for i, (data, labels,renewlabel_list) in enumerate(train_loader, resume_step):
        print(renewlabel_list)
        data, labels = data.cuda(), labels.cuda()
        # print(data.size(),labels.size())
        data_ed = datetime.now()

        '''# zjz-2025-1-24-version3
        # 原来路径 /root/prostate-check-zjz/Medical-SAM2-zjz/data/USVideo_final/Train/xxx/
        # 标签不为0
        if labels[0] != 0 :
            #print("Frame shape:", data.shape)
            #shape [32, 3, 8, 224, 224] 
            # 所有frames存到medsam2文件中按照test格式排版
            base_dir = '/root/prostate-check-zjz/Medical-SAM2-zjz/Test-Medsam2'
            num_existing_folders = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

            i = num_existing_folders + 1
            new_test_folder = os.path.join(base_dir, f'Test-{i}')
            os.makedirs(new_test_folder, exist_ok=True)
            
            test_subfolder = os.path.join(new_test_folder, 'Test')
            os.makedirs(test_subfolder, exist_ok=True)

            for batch_idx, batch in enumerate(data):  # 遍历每个批次
                for frame_idx in range(batch.shape[1]):  # 遍历每一帧（总共 8 帧）
                    frame = batch[:, frame_idx]  # 提取每一帧，形状为 [3, 224, 224]
                    # print("Frame shape:", frame.shape)
                    #  Frame shape: torch.Size([3, 224, 224])
                    # 每一帧的文件夹路径
                    idx = frame_idx+max(0,batch_idx-1)*8
                    frame_folder = os.path.join(test_subfolder, f'idx_{idx+1}')
                    os.makedirs(frame_folder, exist_ok=True)
                    
                    img_path = os.path.join(frame_folder, f'idx_{idx+1}_img.png')
                    label_path = os.path.join(frame_folder, f'idx_{idx+1}_label.png')

                    if frame.shape[0] == 3:  # 确保是RGB图像
                        frame_gray = 0.2989 * frame[0].cpu().numpy() + 0.5870 * frame[1].cpu().numpy() + 0.1140 * frame[2].cpu().numpy()
                    else:  # 如果已经是灰度图像
                        frame_gray = frame[0].cpu().numpy()

                    # 将灰度图像转换到[0, 255]范围并转为uint8类型
                    frame_gray = (frame_gray * 255).astype(np.uint8)

                    Image.fromarray(frame_gray).save(img_path)
                    Image.fromarray(frame_gray).save(label_path)
        # zjz-2025-1-24-version3'''

        optimizer.zero_grad()

        assert data.size(0) % args.batch_split == 0
        split_size = data.size(0) // args.batch_split
        hit1, hit5, loss_value = 0, 0, 0

        all_preds = []
        all_labels = []


        for j in range(args.batch_split):
            data_slice = data[split_size * j: split_size * (j + 1)]
            labels_slice = labels[split_size * j: split_size * (j + 1)]

            # print(labels_slice.size())

            '''
            with torch.amp.autocast('cuda', args.fp16):
            '''
            with torch.cuda.amp.autocast(args.fp16):
                logits = model(data_slice)
                loss = criterion(logits, labels_slice)
                # print("log",logits, "lab",labels_slice,loss)
                
            if labels.dtype == torch.long: # no mixup, can calculate accuracy
                pred = logits.topk(1, dim=1)[1]
                pred = pred.view(-1)
                patch_labels = labels_slice.view(-1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(patch_labels.cpu().numpy())

                total_label_1 += (patch_labels==1).sum().item()
                total_label_2 += (patch_labels==2).sum().item()
                total_label_0 += (patch_labels==0).sum().item()

                misclassfied_1_as_0 +=((patch_labels==1)&(pred==0)).sum().item()
                misclassfied_2_as_0 +=((patch_labels==2)&(pred==0)).sum().item()
                misclassfied_0_as_1or2 +=((patch_labels==0)&(pred!=0)).sum().item()

                hit1 += (logits.topk(1, dim=1)[1] == labels_slice.view(-1, 1)).sum().item()
                # hit5 += (logits.topk(5, dim=1)[1] == labels_slice.view(-1, 1)).sum().item()
            loss_value += loss.item() / args.batch_split
            
            loss_scaler.scale(loss / args.batch_split).backward()
        
        loss_scaler.step(optimizer)
        loss_scaler.update()
        lr_sched.step()

        batch_ed = datetime.now()

        if i % args.print_freq == 0:
            sync_tensor = torch.Tensor([loss_value, hit1 / data.size(0), hit5 / data.size(0)]).cuda()
            dist.all_reduce(sync_tensor)
            sync_tensor = sync_tensor.cpu() / dist.get_world_size()
            loss_value, acc1, acc5 = sync_tensor.tolist()

            if len(all_preds) > 0 and len(all_labels) > 0:
                precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            else:
                precision, recall, f1 = 0, 0, 0

            if total_label_1 > 0:
                misclassfied_1_as_0_rate = misclassfied_1_as_0 / total_label_1
            else:
                misclassfied_1_as_0_rate = 0.0

            if total_label_2 > 0:
                misclassfied_2_as_0_rate = misclassfied_2_as_0 / total_label_2
            else:
                misclassfied_2_as_0_rate = 0.0

            if total_label_1 > 0 or total_label_2 > 0:
                misclassfied_12_as_0_rate = (misclassfied_1_as_0 + misclassfied_2_as_0) / (total_label_1 + total_label_2)
            else:
                misclassfied_12_as_0_rate = 0.0

            if total_label_0 > 0:
                misclassfied_0_as_1or2_rate = misclassfied_0_as_1or2 / total_label_0
            else:
                misclassfied_0_as_1or2_rate = 0.0

            print(
                f'{i + 1} steps valid: '
                f'batch_time: {(batch_ed - batch_st).total_seconds():.3f}  '
                f'data_time: {(data_ed - batch_st).total_seconds():.3f}  '
                f'ETA: {(batch_ed - train_st) / (i - resume_step + 1) * (args.num_steps - i - 1)}  |  '
                f'lr: {optimizer.param_groups[0]["lr"]:.6f}  '
                f'loss: {loss_value:.6f}' + (
                    f'  acc1: {acc1 * 100:.2f}%  acc5: {acc5 * 100:.2f}%' if labels.dtype == torch.long else ''
                ) + (
                    f'  precision: {precision * 100:.2f}%  recall: {recall * 100:.2f}%  f1: {f1 * 100:.2f}%' 
                )
            )
            # print(type(misclassfied_1_as_0_rate),misclassfied_1_as_0_rate)
            print(
                f"MisClassification 1 as 0 rate: {misclassfied_1_as_0_rate * 100:.2f}%,"
                f"MisClassification 2 as 0 rate: {misclassfied_2_as_0_rate * 100:.2f}%,"
                f"MisClassification 1 or 2 as 0 rate: {misclassfied_12_as_0_rate * 100:.2f}%,"
                f"MisClassification 0 as 1 or 2 rate: {misclassfied_0_as_1or2_rate * 100:.2f}%."
            )
        
        if (i + 1) % args.eval_freq == 0:
            print('Start model evaluation at step', i + 1)
            model.eval()
            evaluate(model, val_loader)
            model.train()

        if (i + 1) % args.save_freq == 0 and dist.get_rank() == 0:
            checkpoint.save_checkpoint(model, optimizer, lr_sched, loss_scaler, i + 1, args)
        
        batch_st = datetime.now()


# def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
#     tot, hit1, hit5 = 0, 0, 0
#     eval_st = datetime.now()
#     for data, labels in loader:
#         data, labels = data.cuda(), labels.cuda()
#         assert data.size(0) == 1
#         if data.ndim == 6:
#             data = data[0] # now the first dimension is number of views

#         with torch.no_grad():
#             logits = model(data)
#             scores = logits.softmax(dim=-1).mean(dim=0)
#             # print(scores,labels)

#         tot += 1
#         hit1 += (scores.topk(1)[1] == labels).sum().item()
#         # hit5 += (scores.topk(5)[1] == labels).sum().item()

#         if tot % 20 == 0:
#             print(f'[Evaluation] num_samples: {tot}  '
#                   f'ETA: {(datetime.now() - eval_st) / tot * (len(loader) - tot)}  '
#                   f'cumulative_acc1: {hit1 / tot * 100.:.2f}%  '
#                   f'cumulative_acc5: {hit5 / tot * 100.:.2f}%')

#     sync_tensor = torch.LongTensor([tot, hit1, hit5]).cuda()
#     dist.all_reduce(sync_tensor)
#     tot, hit1, hit5 = sync_tensor.cpu().tolist()

#     print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}% ')
#     # print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%, top5={hit5 / tot * 100:.2f}%')

def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader):
    tot, hit1, hit5 = 0, 0, 0
    all_preds = []  # 用来存储所有预测值
    all_labels = []  # 用来存储所有真实标签
    misclassfied_1_as_0 = 0
    misclassfied_2_as_0 = 0
    misclassfied_0_as_1or2 = 0
    total_label_1 = 0
    total_label_2 = 0
    total_label_0 = 0
    eval_st = datetime.now()

    for data, labels in loader:
        data, labels = data.cuda(), labels.cuda()
        assert data.size(0) == 1
        if data.ndim == 6:
            data = data[0]  # now the first dimension is number of views

        with torch.no_grad():
            logits = model(data)
            scores = logits.softmax(dim=-1).mean(dim=0)

        tot += 1
        pred = scores.topk(1)[1]  # 获取预测的标签
        all_preds.extend(pred.cpu().numpy())  # 累积预测结果
        all_labels.extend(labels.cpu().numpy())  # 累积真实标签

        total_label_1 += (labels==1).sum().item()
        total_label_2 += (labels==2).sum().item()
        total_label_0 += (labels==0).sum().item()

        misclassfied_1_as_0 +=((labels==1)&(pred==0)).sum().item()
        misclassfied_2_as_0 +=((labels==2)&(pred==0)).sum().item()
        misclassfied_0_as_1or2 +=((labels==0)&(pred!=0)).sum().item()


        hit1 += (pred == labels).sum().item()
        # hit5 += (scores.topk(5)[1] == labels).sum().item()

        if tot % 20 == 0:
            print(f'[Evaluation] num_samples: {tot}  '
                  f'ETA: {(datetime.now() - eval_st) / tot * (len(loader) - tot)}  '
                  f'cumulative_acc1: {hit1 / tot * 100.:.2f}%  '
                  f'cumulative_acc5: {hit5 / tot * 100.:.2f}%')

    # 计算 Precision, Recall 和 F1
    if len(all_preds) > 0 and len(all_labels) > 0:
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    else:
        precision, recall, f1 = 0, 0, 0
    
    if total_label_1 > 0 or total_label_2 > 0:
        misclassfied_1_as_0_rate = misclassfied_1_as_0 / total_label_1
        misclassfied_2_as_0_rate = misclassfied_2_as_0 / total_label_2
        misclassfied_12_as_0_rate = (misclassfied_1_as_0 + misclassfied_2_as_0) / (total_label_1 + total_label_2)
        misclassfied_0_as_1or2_rate = misclassfied_0_as_1or2 / total_label_0
    else:
        misclassfied_1_as_0_rate = 0
        misclassfied_2_as_0_rate = 0
        misclassfied_12_as_0_rate = 0
        misclassfied_0_as_1or2_rate = 0

    sync_tensor = torch.LongTensor([tot, hit1, hit5]).cuda()
    dist.all_reduce(sync_tensor)
    tot, hit1, hit5 = sync_tensor.cpu().tolist()

    print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%  Recall: {recall * 100:.2f}%  F1: {f1 * 100:.2f}%')
    print(
        f"MisClassification 1 as 0 rate: {misclassfied_1_as_0_rate * 100:.2f}%,"
        f"MisClassification 2 as 0 rate: {misclassfied_2_as_0_rate * 100:.2f}%,"
        f"MisClassification 1 or 2 as 0 rate: {misclassfied_12_as_0_rate * 100:.2f}%,"
        f"MisClassification 0 as 1 or 2 rate: {misclassfied_0_as_1or2_rate * 100:.2f}%."
    )
    # 如果需要Top-5准确率，可以解除下面的注释
    # print(f'Accuracy on validation set: top1={hit1 / tot * 100:.2f}%, top5={hit5 / tot * 100:.2f}%')

if __name__ == '__main__': main()
