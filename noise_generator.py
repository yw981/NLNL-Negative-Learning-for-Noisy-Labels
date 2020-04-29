import args
import os
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
import random
import pickle

random.seed(10)
opt = args.args()

try:
    os.makedirs('noise/%s' % (opt.noise_type))
except OSError:
    pass
###################################################################################################
if opt.dataset == 'cifar10_wo_val':
    num_classes = 10
else:
    print('There exists no data')

trainset = dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), transform=transforms.ToTensor())
# trainset.imgs是45000个元素的list，每个元素是（str,int）的tuple，图片路径和标签
# clean_labels是标签，0~9按顺序，每个4500张
clean_labels = np.array(trainset.imgs)[:, 1]

for n in range(2):
    # 没用？
    # trainset = dset.ImageFolder(root='{}/{}/train'.format(opt.dataroot, opt.dataset), transform=transforms.ToTensor())

    noisy_idx = []
    for c in range(num_classes):
        # np.where只有一个参数，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。
        # 这里的坐标以tuple的形式给出，通常原数组有多少维，此处返回（1，）
        # 取样个数 len(trainset.imgs) * (n * 0.1 / num_classes))) = 450 * 10% * n
        noisy_idx.extend(random.sample(list(np.where(clean_labels.astype(int) == c)[0]),
                                       int(len(trainset.imgs) * (n * 0.1 / num_classes))))
        # print(noisy_idx)

    # 把imgs每个元素从（str,int）的tuple转成list
    trainset.imgs_temp = np.empty_like(trainset.imgs)  # to change tuple to list
    trainset.imgs_temp = [list(trainset.imgs[i]) for i in range(len(trainset.imgs))]
    trainset.imgs = trainset.imgs_temp
    # print(trainset.imgs[0])

    for i in noisy_idx:

        if 'symm_exc' in opt.noise_type:
            # random.sample从数组里取不重复的k个元素
            # i是要生成noise标签的图id，trainset.imgs[i][1]是标签
            # 构造不含正确标签的数组，从(0, trainset.imgs[i][1])和(trainset.imgs[i][1] + 1, num_classes)
            trainset.imgs[i][1] = \
                random.sample(
                    list(range(0, trainset.imgs[i][1])) + list(range(trainset.imgs[i][1] + 1, num_classes)),
                    1)[0]

        elif 'asymm' in opt.noise_type:
            if opt.dataset == 'cifar10_wo_val':
                if trainset.imgs[i][1] == 9:
                    trainset.imgs[i][1] = 1
                elif trainset.imgs[i][1] == 2:
                    trainset.imgs[i][1] = 0
                elif trainset.imgs[i][1] == 3:
                    trainset.imgs[i][1] = 5
                elif trainset.imgs[i][1] == 5:
                    trainset.imgs[i][1] = 3
                elif trainset.imgs[i][1] == 4:
                    trainset.imgs[i][1] = 7

    noisy_labels = np.array(trainset.imgs)[:, 1]
    # 打印noise占正确的（全部）比例
    print(float(np.sum(clean_labels != noisy_labels)) / len(clean_labels))
    with open('noise/%s/train_labels_n%02d_%s' % (opt.noise_type, n * 10, opt.dataset), 'wb') as fp:
        pickle.dump(noisy_labels, fp)

    for k in range(num_classes):  # Checking class-wise noise ratio
        clean_int = clean_labels.astype(int)
        noisy_int = noisy_labels.astype(int)
        print(k, len(clean_int[clean_int == k]),
              float(np.sum(clean_int[clean_int == k] != noisy_int[clean_int == k])) / len(clean_int[clean_int == k]))

    leng = int(len(trainset) / num_classes)
    for k in range(num_classes):  # Checking class-wise noise
        print(noisy_labels[leng * k:leng * k + 15])
