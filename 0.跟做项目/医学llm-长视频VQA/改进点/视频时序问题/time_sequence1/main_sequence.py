'''
Description: Main function of PitVQA-Net model
Paper: PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery
Author: Runlong He, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, 
        Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam
Lab: Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS), UCL
Acknowledgement : Code adopted from the official implementation of 
                  Huggingface Transformers (https://github.com/huggingface/transformers)
                  and Surgical-GPT (https://github.com/lalithjets/SurgicalGPT).
'''

from datetime import datetime
import os
import torch
import argparse
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import random

from torch import nn
from utils_sequence import save_clf_checkpoint, adjust_learning_rate, calc_acc, calc_precision_recall_fscore
from torch.utils.data import DataLoader

from dataloader_sequence import EndoVis18VQAGPTClassification, Pit24VQAClassification
from model_sequence import PitVQANet

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(train_dataloader, model, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0.0
    label_true = None
    label_pred = None
    label_score = None

    for i, (_, images, videos ,questions, labels) in enumerate(train_dataloader, 0):
        # labels
        labels = labels.to(device)
        outputs = model(image=images.to(device),video = videos.to(device), question=questions)  # questions is a tuple
        loss = criterion(outputs, labels)  # calculate loss
        optimizer.zero_grad()
        loss.backward()  # calculate gradient
        optimizer.step()  # update parameters

        # print statistics
        total_loss += loss.item()

        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        if label_true is None:  # accumulate true labels of the entire training set
            label_true = labels.data.cpu()
        else:
            label_true = torch.cat((label_true, labels.data.cpu()), 0)
        if label_pred is None:  # accumulate pred labels of the entire training set
            label_pred = predicted.data.cpu()
        else:
            label_pred = torch.cat((label_pred, predicted.data.cpu()), 0)
        if label_score is None:
            label_score = scores.data.cpu()
        else:
            label_score = torch.cat((label_score, scores.data.cpu()), 0)

        # print('label_true:',label_true)
        # print('label_pred:',label_pred)

    # loss and acc
    # print('label_true:',label_true)
    # print('label_pred:',label_pred)

    acc = calc_acc(label_true, label_pred)
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print(f'Train: epoch: {epoch} loss: {total_loss} | Acc: {acc} | '
          f'Precision: {precision} | Recall: {recall} | F1 Score: {f_score}')
    
    return acc


def validate(val_loader, model, criterion, epoch, device):
    model.eval()
    total_loss = 0.0
    label_true = None
    label_pred = None

    with torch.no_grad():
        for i, (file_name, images, questions, labels) in enumerate(val_loader, 0):
            # label
            labels = labels.to(device)
        
            # model forward pass
            outputs = model(image=images.to(device), question=questions)

            # loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            label_true = labels.data.cpu() if label_true is None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred is None else torch.cat((label_pred, predicted.data.cpu()), 0)

    acc = calc_acc(label_true, label_pred)
    precision, recall, f_score = calc_precision_recall_fscore(label_true, label_pred)
    print(f'Test: epoch: {epoch} test loss: {total_loss} | test acc: {acc} | '
          f'test precision: {precision} | test recall: {recall} | test F1: {f_score}')
    return acc, precision, recall, f_score


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_arg():
    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=60,                   help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=40,                   help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                    help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--lr',             type=float, default=0.00002,              help=' 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default='checkpoints/pitvqa_testing_',    help='m18/c80')
    parser.add_argument('--question_len',   default=25,                               help='25')
    parser.add_argument('--random_seed',    type=int,   default=42,                   help='random seed')
    parser.add_argument('--dataset',                    default='endo18',             help='endo18/pit24')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_arg()

    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(f'./checkpoints/{time_str}', exist_ok=True)
    args.checkpoint_dir = f'checkpoints/{time_str}/pitvqa_testing_'

    seed_everything(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0

    train_dataloader = None
    val_dataloader = None
    num_class = 0
    if args.dataset == 'endo18':
        num_class = 18
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]  # 11
        val_seq = [1, 5, 16]
        print(f'current train seq: {train_seq}')
        print(f'current val seq: {val_seq}')

        folder_head = '/SAN/medic/CARES/mobarak/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Classification/*.txt'

        # dataloader
        train_dataset = EndoVis18VQAGPTClassification(train_seq, folder_head, folder_tail)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataset = EndoVis18VQAGPTClassification(val_seq, folder_head, folder_tail)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    elif args.dataset == 'pit24':
        num_class = 59
        train_seq = ['01', '03', '04', '05', '07', '08', '09', '10', '11', '14',
                     '15', '16', '17', '18', '19', '20', '21', '22', '23', '25']
        val_seq = ['02', '06', '12', '13', '24']
        print(f'current train seq: {train_seq}')
        print(f'current val seq: {val_seq}')
        
        # /home/test/PitVQA-main/PitVQA_dataset/qa-classification/video_04
        folder_head = '/home/test/PitVQA-main/PitVQA_dataset/qa-classification/video_'
        folder_tail = '/*.txt'

        # dataloader
        train_dataset = Pit24VQAClassification(train_seq, folder_head, folder_tail)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        val_dataset = Pit24VQAClassification(val_seq, folder_head, folder_tail)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = PitVQANet(num_class=num_class)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    print('Start training.')
    for epoch in range(start_epoch, args.epochs+1):

        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # train
        train_acc = train(train_dataloader=train_dataloader, model=model, criterion=criterion,
                          optimizer=optimizer, epoch=epoch, device=device)
        # validation
        test_acc, test_precision, test_recall, test_f_score = validate(val_loader=val_dataloader, model=model,
                                                                       criterion=criterion, epoch=epoch, device=device)

        if test_f_score >= best_results[0]:
            print('Best Epoch:', epoch)
            epochs_since_improvement = 0
            best_results[0] = test_f_score
            best_epoch[0] = epoch
            save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement,
                                model, optimizer, best_results[0], final_args=None)
    print('End training.')
