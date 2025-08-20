import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch_dct as dct

from losses import LogitLoss
import time

#
#     Arguments:
#         model(nn.Module): model to attack.
#         eps(float): maximum perturbation.(Default: 0.01)
#     Shape:
#         - images::math: `(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`.
#         - labels::math: `(N)`
#         - output::math: `(N, C, H, W)`.
#

class Base(object):
    def __init__(self, loss_fn, steps, random_start=False, alpha=2/255):
        self.steps = steps
        self.random_start = random_start
        self.alpha = alpha
        if loss_fn == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'Logit':
            self.loss_fn = LogitLoss()

        self.begin_time = time.time()

    def _target_layer(self, model_name, depth):
        '''
        'resnet18', 'vgg11',   depth: [1, 2, 3, 4]
        根据给定的模型名称（model_name）和深度（depth），从预训练的深度学习模型中获取特定的层
        '''
        if model_name == 'resnet18':
            return getattr(self.model, 'layer{}'.format(depth))[-1]
        elif model_name == 'vgg11':
            return getattr(self.model, 'conv_layer{}'.format(depth))

    def perturb(self, images, ori_labels):
        raise NotImplementedError

class ours_FGSM(Base):
    def __init__(self, model_name, depth, coef, model, loss_fn, steps, epsilon=0.03, num_copies=10, num_block=3):
        super(ours_FGSM, self).__init__(loss_fn, steps)
        device = None

        self.model_name = model_name
        self.depth = depth
        self.coef = coef
        self.model = model
        self.epsilon = epsilon

        self.device = next(model.parameters()).device if device is None else device
        self.num_copies = num_copies
        self.num_block = num_block

        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip,
                   self.rotate180, self.scale, self.add_noise, self.dct, self.drop_out]
        self._register_forward()

    def _register_forward(self):  # 模型在特定层捕获输出
        '''
        'resnet18',  'vgg11'
        '''
        self.activations = {'ori': None, 'adv': None}  # 保存原图像和对抗图像的特征图

        def forward_hook(module, input, output):

            if self.is_adv == 2:
                self.activations['adv'] = output
            elif self.is_adv == 1:
                self.activations['ori'] = output

            return None

        target_layer = self._target_layer(self.model_name, self.depth)
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
        else:
            print(f"Warning: Target layer for model {self.model_name} at depth {self.depth} not found.")

    def sia(self, inputs):
        inputs = torch.cat([self.blocktransform(inputs) for _ in range(self.num_copies)], dim=0)
        return inputs

    def blocktransform(self, x, choice=-1):  # 分块变换

        _, c, w, h = x.shape
        y_axis = [0, ] + np.random.choice(list(range(1, h)), self.num_block - 1, replace=False).tolist() + [h, ]
        x_axis = [0, ] + np.random.choice(list(range(1, w)), self.num_block - 1, replace=False).tolist() + [w, ]
        c_axis = [0, ] + np.random.choice(list(range(1, c)), self.num_block - 1, replace=False).tolist() + [c, ]
        y_axis.sort()
        x_axis.sort()
        c_axis.sort()

        x_copy = x.clone()
        # 随机选择变换
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                for k, idx_c in enumerate(c_axis[1:]):
                    chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)

                    x_copy[:, c_axis[k]:idx_c, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
                        x_copy[:, c_axis[k]:idx_c, x_axis[i]:idx_x, y_axis[j]:idx_y]
                    )

        return x_copy

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x, min_scale=0.3):
        r = torch.rand(1).item() * (1 - min_scale) + min_scale
        return r * x

    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x

    def dct(self, x):  # 离散余弦变换

        dctx = dct.dct_2d(x)
        _, _, w, h = dctx.shape  # 模拟丢弃高频分量
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        # dctx[:, :, -low_w:, -low_h:] = 0
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx
        idctx = dct.idct_2d(dctx)
        return idctx

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def drop_out(self, x):
        return F.dropout2d(x, p=0.15, training=True)

    def compute_channel_weights(self, feature):  # 计算通道权重
        channel_variances = feature.var(dim=[0, 2, 3], unbiased=False)

        total_variance = channel_variances.sum()
        channel_weights = channel_variances / total_variance

        return channel_weights

    def perturb(self, images, ori_labels):

        adv_images = images.clone().detach().to(self.device)
        ori_labels = ori_labels.clone().detach().to(self.device)
        batch_size = adv_images.shape[0]

        self.is_adv = 1
        self.model(adv_images)

        ori_feature = 1.2 * self.activations["ori"]  #Enlarge the feature values of the original example
        channel_weights_ori = self.compute_channel_weights(ori_feature)

        delta = torch.zeros_like(adv_images, requires_grad=True).to(self.device)

        for itr in range(self.steps):
            augmented_inputs = self.sia(adv_images+delta)

            self.is_adv = 2
            logits = self.model(augmented_inputs)
            loss_label = torch.cat([ori_labels] * int(self.num_copies))
            classifier_loss = self.loss_fn(logits, loss_label)

            fs_losses=[]
            for sia_ind in range(self.num_copies):
                advbatch_features = self.activations['adv'][sia_ind * batch_size:(sia_ind + 1) * batch_size]

                P_i = torch.sum((advbatch_features - ori_feature) ** 2, axis=(0, 2, 3))
                Q_i = torch.sum(torch.abs(advbatch_features) + torch.abs(ori_feature) + 1e-8,axis=(0, 2, 3))
                fs_loss = torch.sum(channel_weights_ori * (P_i / Q_i))
                fs_losses.append(fs_loss)

            fs_loss_total = torch.mean(torch.stack(fs_losses))  # Weighted feature divergence loss

            loss = fs_loss_total+ self.coef *classifier_loss   # Total loss

            loss.backward()

            grad_c = delta.grad.clone()

            grad_a = grad_c.clone()

            delta.grad.zero_()
            delta.data = delta.data + self.epsilon * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.data = ((adv_images + delta).clamp(0, 1)) - adv_images

        advi = adv_images + delta
        return advi

class ours_MI(Base):

    def __init__(self, model_name, depth, coef,model, loss_fn, steps, epsilon=0.03, num_copies=10, num_block=3):
        super(ours_MI, self).__init__(loss_fn, steps)
        device = None

        self.model_name = model_name
        self.depth = depth
        self.coef = coef
        self.model = model

        self.epsilon = epsilon

        self.device = next(model.parameters()).device if device is None else device
        self.num_copies = num_copies
        self.num_block = num_block

        self.op = [self.resize, self.vertical_shift, self.horizontal_shift, self.vertical_flip,
                   self.horizontal_flip,
                   self.rotate180, self.scale, self.add_noise, self.dct, self.drop_out]
        self._register_forward()

    def _register_forward(self):  # 允许模型在特定层捕获输出
        '''
        'resnet18',  'vgg11'
        '''
        self.activations = {'ori': None, 'adv': None}

        def forward_hook(module, input, output):

            if self.is_adv == 2:
                self.activations['adv'] = output
            elif self.is_adv == 1:
                self.activations['ori'] = output

            return None

        target_layer = self._target_layer(self.model_name, self.depth)
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
        else:
            print(f"Warning: Target layer for model {self.model_name} at depth {self.depth} not found.")

    def sia(self, inputs):
        inputs = torch.cat([self.blocktransform(inputs) for _ in range(self.num_copies)], dim=0)
        return inputs

    def blocktransform(self, x, choice=-1):  # 分块变换

        _, c, w, h = x.shape
        y_axis = [0, ] + np.random.choice(list(range(1, h)),  self.num_block - 1, replace=False).tolist() + [h, ]
        x_axis = [0, ] + np.random.choice(list(range(1, w)),  self.num_block - 1, replace=False).tolist() + [w, ]
        c_axis = [0, ] + np.random.choice(list(range(1, c)), self.num_block - 1, replace=False).tolist() + [c, ]
        y_axis.sort()
        x_axis.sort()
        c_axis.sort()

        x_copy = x.clone()

        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                for k, idx_c in enumerate(c_axis[1:]):
                    chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)

                    x_copy[:, c_axis[k]:idx_c, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
                        x_copy[:, c_axis[k]:idx_c, x_axis[i]:idx_x, y_axis[j]:idx_y]
                    )
        return x_copy

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x, min_scale=0.3):
        r = torch.rand(1).item() * (1 - min_scale) + min_scale
        return r * x

    def resize(self, x):
        """
        Resize the input
        """
        _, _, w, h = x.shape
        scale_factor = 0.8
        new_h = int(h * scale_factor) + 1
        new_w = int(w * scale_factor) + 1
        x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(0, 1)
        return x

    def dct(self, x):  # 离散余弦变换

        dctx = dct.dct_2d(x)
        _, _, w, h = dctx.shape  # 模拟丢弃高频分量
        low_ratio = 0.4
        low_w = int(w * low_ratio)
        low_h = int(h * low_ratio)
        dctx[:, :, -low_w:, :] = 0
        dctx[:, :, :, -low_h:] = 0
        dctx = dctx
        idctx = dct.idct_2d(dctx)
        return idctx

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)

    def compute_channel_weights(self, feature):  #计算通道权重

        channel_variances = feature.var(dim=[0, 2, 3], unbiased=False)

        total_variance = channel_variances.sum()
        channel_weights = channel_variances / total_variance

        return channel_weights

    def perturb(self, images, ori_labels):

        adv_images = images.clone().detach().to(self.device)
        ori_labels = ori_labels.clone().detach().to(self.device)
        batch_size = adv_images.shape[0]

        self.is_adv = 1
        self.model(adv_images)

        ori_feature = 1.2 * self.activations["ori"]
        channel_weights_ori = self.compute_channel_weights(ori_feature)

        delta = torch.zeros_like(adv_images, requires_grad=True).to(self.device)

        grad_pre = 0
        for itr in range(self.steps):
            augmented_inputs = self.sia(adv_images + delta)

            self.is_adv = 2
            logits = self.model(augmented_inputs)
            loss_label = torch.cat([ori_labels] * int(self.num_copies))
            classifier_loss = self.loss_fn(logits, loss_label)

            fs_losses = []
            for sia_ind in range(self.num_copies):

                advbatch_features = self.activations['adv'][sia_ind * batch_size:(sia_ind + 1) * batch_size]
                P_i = torch.sum((advbatch_features - ori_feature) ** 2, axis=(0, 2, 3))
                Q_i = torch.sum(torch.abs(advbatch_features) + torch.abs(ori_feature) + 1e-8,axis=(0, 2, 3))
                fs_loss = torch.sum(channel_weights_ori * (P_i / Q_i))
                fs_losses.append(fs_loss)

            fs_loss_total = torch.mean(torch.stack(fs_losses))  #加权特征远离损失

            loss = fs_loss_total + self.coef * classifier_loss  # 总损失

            loss.backward()

            grad_c = delta.grad.clone()

            grad_a = grad_c / (torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1e-8) + 1 * grad_pre
            grad_pre = grad_a

            delta.grad.zero_()
            delta.data = delta.data + self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.data = ((adv_images + delta).clamp(0, 1)) - adv_images

        advi = adv_images + delta
        return advi