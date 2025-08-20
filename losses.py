import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitLoss(nn.Module):
    def __init__(self, ):
        super(LogitLoss, self).__init__()

    def forward(self, logits, labels):
        real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
        #从logits通常是 [batch_size, num_classes]中根据 labels（形状为 [batch_size] 的张量）选取每个样本对应类别的 logit 值。
        logit_dists = ( -1 * real) #计算了选取出的 logit 值的负值
        loss = logit_dists.mean() #负对数值的平均值作为损失
        return loss

