'''
Description: Utils implementation of PitVQA-Net model
Paper: PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery
Author: Runlong He, Mengya Xu, Adrito Das, Danyal Z. Khan, Sophia Bano, 
        Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam
Lab: Wellcome/EPSRC Centre for Interventional and Surgical Sciences (WEISS), UCL
Acknowledgement : Code adopted from the official implementation of 
                  Huggingface Transformers (https://github.com/huggingface/transformers)
                  and Surgical-GPT (https://github.com/lalithjets/SurgicalGPT).
'''

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, metrics, is_best, final_args):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update model's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'metrics': metrics,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)


def save_clf_checkpoint(checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, Acc, final_args):
    """
    Saves model checkpoint.
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'Acc': Acc,
             'model': model,
             'optimizer': optimizer,
             'final_args': final_args}
    filename = checkpoint_dir + 'Best.pth.tar'
    torch.save(state, filename)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def calc_acc(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    return acc


def calc_classwise_acc(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    classwise_acc = matrix.diagonal()/matrix.sum(axis=1)
    return classwise_acc


def calc_map(y_true, y_scores):
    mAP = average_precision_score(y_true, y_scores, average=None)
    return mAP


def calc_precision_recall_fscore(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=1)
    return (precision, recall, fscore)


def compare_files(file1, file2):

    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        # 逐行读取并比较
        for line1, line2 in zip(f1, f2):
            if line1 != line2:
                return False
        # 检查文件长度是否一致（可能有文件较长，另一个文件较短的情况）
        return f1.readline() == f2.readline()
    

def get_95_percent(file_path):
    # Reloading data from the provided file
    with open(file_path, 'r') as file:
        numbers = file.readlines()

    # Convert to integers
    numbers = list(map(int, numbers))

    # Calculate the 95th percentile of the frame numbers
    percentile_95 = np.percentile(numbers, 95)

    percentile_95=int(percentile_95)

    return (percentile_95)


def padding_and_slicing(video,frames_num):
    if len(video) < frames_num :
        while (len(video)< frames_num):
            video.append(video[0])
    else :
        video = (video[:frames_num])  # 裁剪

    return video

def compare_images(image_path1, image_path2):
    # 打开两张图片
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    
    # 确保两张图片大小相同
    if img1.size != img2.size:
        raise ValueError("The images must have the same dimensions.")
    
    # 将图片转为 RGB 模式并转换为 NumPy 数组
    img1_array = np.array(img1.convert('RGB'))
    img2_array = np.array(img2.convert('RGB'))
    
    # 计算像素差异，使用欧氏距离计算
    diff = np.linalg.norm(img1_array - img2_array, axis=2)  # 计算每个像素的差异
    
    # 总差异：可以使用整个图像的平均差异或者总和
    total_diff = np.sum(diff)  # 总差异（像素值差的总和）

    return total_diff