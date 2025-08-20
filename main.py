import numpy as np
import argparse
import torch
from scipy import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

import resnet
from vgg11 import VGG11

import attack


def loadData():
    DataPath = '900(1000)_PaviaU01/paviaU.mat'
    TRPath = '900(1000)_PaviaU01/TRLabel.mat'
    TSPath = '900(1000)_PaviaU01/TSLabel.mat'


    # load data
    Data = io.loadmat(DataPath)
    TrLabel = io.loadmat(TRPath)
    TsLabel = io.loadmat(TSPath)


    Data = Data['paviaU']
    Data = Data.astype(np.float32)
    TrLabel = TrLabel['TRLabel']
    TsLabel = TsLabel['TSLabel']


    return Data, TrLabel, TsLabel


def createPatches(X, y, windowSize):
    [m, n, l] = np.shape(X)
    temp = X[:, :, 0]
    pad_width = np.floor(windowSize / 2)
    pad_width = np.int_(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    [m2, n2] = temp2.shape
    x2 = np.empty((m2, n2, l), dtype='float32')

    for i in range(l):
        temp = X[:, :, i]
        pad_width = np.floor(windowSize / 2)
        pad_width = np.int_(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2[:, :, i] = temp2

    [ind1, ind2] = np.where(y != 0)
    TrainNum = len(ind1)
    patchesData = np.empty((TrainNum, l, windowSize, windowSize), dtype='float32')
    patchesLabels = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch = x2[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch = np.reshape(patch, (windowSize * windowSize, l))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (l, windowSize, windowSize))
        patchesData[i, :, :, :] = patch
        patchlabel = y[ind1[i], ind2[i]]
        patchesLabels[i] = patchlabel

    return patchesData, patchesLabels


def Normalize(dataset):
    [m, n, b] = np.shape(dataset)
    # change to [0,1]
    for i in range(b):
        _range = np.max(dataset[:, :, i]) - np.min(dataset[:, :, i])
        dataset[:, :, i] = (dataset[:, :, i] - np.min(dataset[:, :, i])) / _range

    return dataset


def TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize):
    Data_TR, TR_gt_M = createPatches(Data, TR_gt, windowSize)
    Data_TS, TS_gt_M = createPatches(Data, TS_gt, windowSize)

    # change to the input type of PyTorch
    Data_TR = torch.from_numpy(Data_TR)
    Data_TS = torch.from_numpy(Data_TS)

    TrainLabel = torch.from_numpy(TR_gt_M) - 1
    TrainLabel = TrainLabel.long()
    TestLabel = torch.from_numpy(TS_gt_M) - 1
    TestLabel = TestLabel.long()

    return Data_TR, Data_TS, TrainLabel, TestLabel

def str2bool(v):
    return v.lower() in ('true', '1')


def main():
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--white_box', type=str, default='resnet18', help='resnet18, vgg11')
    parser.add_argument('--attacks', type=str, default='ours_MI',
                    help='class in attack:  ours_FGSM, ours_MI ')
    parser.add_argument('--loss_fn', type=str, default='CE', help='cross-entropy loss')
    parser.add_argument('--steps', type=int, default=20, help=' iteration steps (default: 20)')

    parser.add_argument('--num_copies', default=10, type=int, help='number of copies (default: 10)')
    parser.add_argument('--num_block', default=3, type=int, help='number of shuffled blocks (default: 3)')

    parser.add_argument('--fsl_coef', type=float, default=0.03,
                        help='the coefficient of cross-entropy classification loss (default: 0.03)')
    parser.add_argument('--depth', type=int, default=3, help='the layer used to extract features (default: 3)')

    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epsilon', type=float, default=0.01, metavar='LR',
                        help='adversarial rate (default: 0.01,0.03)')

    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Whether to train or test the model')
    parser.add_argument('--epochs', type=int, default=300,
                        help='# of epochs to train for')
    parser.add_argument('--init_lr', type=float, default=1e-3,
                        help='Initial learning rate value')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='value of weight dacay for regularization')
    parser.add_argument('--use_gpu', type=str2bool, default=False,
                        help="Whether to run on the GPU")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save_path', default='./checkpoint/', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    args = parser.parse_args()

    print("CUDA Available: ", torch.cuda.is_available())
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    Data, TR_gt, TS_gt = loadData()
    [m, n, b] = np.shape(Data) #[610,340,103]
    Classes = len(np.unique(TR_gt)) - 1
    print(Classes)

    Data = Normalize(Data)
    # ==================================================================================================================
    windowSize = 32

    sqr = np.sqrt(windowSize * windowSize * b)

    pretrained_model = "900(1000)_PaviaU01/Train/net_resnet18.pkl"
    pre_model = resnet.ResNet18()
    # pretrained_model ="900(1000)_PaviaU01/Train/net_vgg11.pkl"
    # pre_model = VGG11()
    pre_model = pre_model.to(device)
    pre_model.load_state_dict(torch.load(pretrained_model))
    pre_model.eval()
    for param in pre_model.parameters():
        param.requires_grad = False

    print('Training window size:', windowSize)

    [Data_tr, Data_ts, TrainLabel, TestLabel] = TrainFuc(args, device, Data, TR_gt, TS_gt, windowSize)

    for epoch in range(1):

        part = args.test_batch_size
        number = len(TestLabel) // part

        pred_y_adv = np.empty((len(TestLabel)), dtype='float32')

        adversor = getattr(attack, args.attacks)(args.white_box, args.depth, args.fsl_coef,  pre_model,
                                                     loss_fn=args.loss_fn, steps=args.steps, epsilon=args.epsilon,
                                                     num_copies=args.num_copies, num_block=args.num_block)
        # Test_advSample
        num_correct_adv = 0
        for i in tqdm(range(number)):
            tempdata = Data_ts[i * part:(i + 1) * part, :, :].to(device)
            TestLabel_1 = TestLabel[i * part:(i + 1) * part].to(device)

            x_adv = adversor.perturb(tempdata, TestLabel_1)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv += torch.sum(out_C == TestLabel_1, 0)

            temp_C = pre_model(x_adv)
            temp_CC = torch.max(temp_C, 1)[1].squeeze()
            pred_y_adv[i * part:(i + 1) * part] = temp_CC.cpu()

            torch.cuda.empty_cache()

            del tempdata, TestLabel_1, x_adv, out_C, temp_C, temp_CC,

        if (i + 1) * part < len(TestLabel):
            tempdata = Data_ts[(i + 1) * part:len(TestLabel), :, :].to(device)
            TestLabel_1 = TestLabel[(i + 1) * part:len(TestLabel)].to(device)

            x_adv = adversor.perturb(tempdata, TestLabel_1)

            out_C = torch.argmax(pre_model(x_adv), 1)
            num_correct_adv += torch.sum(out_C == TestLabel_1, 0)

            temp_C = pre_model(x_adv)
            temp_CC = torch.max(temp_C, 1)[1].squeeze()
            pred_y_adv[(i + 1) * part:len(TestLabel)] = temp_CC.cpu()

            torch.cuda.empty_cache()

            del tempdata, TestLabel_1, x_adv, out_C, temp_C, temp_CC

        print(f'num_correct_adv: ', num_correct_adv)
        print(f'accuracy of adv test set: %.5f\n' % (num_correct_adv.item() / len(TestLabel)))

        # Test_adv
        Classes = np.unique(TestLabel)
        EachAcc_adv = np.empty(len(Classes))

        for i in range(len(Classes)):
            cla = Classes[i]
            right = 0
            sum = 0

            for j in range(len(TestLabel)):
                if TestLabel[j] == cla:
                    sum += 1
                if TestLabel[j] == cla and pred_y_adv[j] == cla:
                    right += 1

            EachAcc_adv[i] = right.__float__() / sum.__float__()
        print(EachAcc_adv)


if __name__=='__main__':

    main()
