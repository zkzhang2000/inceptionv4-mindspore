import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import os
from lr_generator import get_lr
from CrossEntropy import CrossEntropy
import argparse
from inception_A import inception_A
from inception_B import inception_B
import numpy as np
from inception_C import inception_C
from network import Stem
from reduction_A import reduction_A
from reduction_B import reduction_B
from reduction_C import reduction_C
import mindspore.dataset as ds
from mindspore import context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
import os
import urllib.request
from urllib.parse import urlparse
import gzip
import argparse
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
# import mindspore.dataset.transforms.vision.c_transforms as CV
# import mindspore.dataset.transforms.c_transforms as C
# from mindspore.dataset.transforms.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore.common import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode
from config import config
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from dataloader import create_dataset

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute')
parser.add_argument('--device_num', type=int, default=8, help='Device num.')
parser.add_argument('--do_train', type=str, default='1', help='Do train or not.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default=None, help='Location of training outputs.')
parser.add_argument('--check_point',default='/',help='checkpoint you need')
opt = parser.parse_args()


from make_dict import dict_need

dict_need=dict_need

class InceptionV4(nn.Cell):
    def __init__(self):
        super().__init__()
        self.Stem = Stem(3)
        self.inception_A = inception_A(384)
        self.reduction_A = reduction_A(384)
        self.inception_B = inception_B(1024)
        self.reduction_B = reduction_B(1024)
        self.inception_C = inception_C(1536)
        self.avgpool = nn.AvgPool2d(8)

        #### reshape成2维
        self.dropout = nn.Dropout(0.8)
        self.linear = nn.Dense(1536, 2388)

    def construct(self, x):
        x = self.Stem(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.reduction_A(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.reduction_B(x)
        x = self.inception_C(x)
        x = self.inception_C(x)
        x = self.inception_C(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = nn.Flatten()(x)
        x = self.linear(x)
        return x

    def generate_inception_module(self, input_channels, output_channels, block_num, block):
        if block == 1:
            layers = nn.SequentialCell([inception_A(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_A(input_channels), layers)
                input_channels = output_channels

        if block == 2:
            layers = nn.SequentialCell([inception_B(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_B(input_channels), layers)
                input_channels = output_channels

        if block == 3:
            layers = nn.SequentialCell([inception_C(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_C(input_channels), layers)
                input_channels = output_channels

        return layers



#########################################
def weight_variable():
    """Weight initial."""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Conv layer weight initial."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """Fc layer weight initial."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)




def ans():
    context.set_context(mode=context.GRAPH_MODE)
    net = InceptionV4()
    print("start")
    ds = create_dataset('./data/train', True, batch_size=config.batch_size)
    lr = 0.005
    optt = nn.Momentum(net.trainable_params(), lr, momentum=0.9)
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_ipv4",directory='./new_check/', config=config_ck)
    # group layers into an object with training and evaluation features
    net_loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=0.0, lr_max=config.lr_max,
                       warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=config.batch_size,
                       lr_decay_mode='cosine'))

    optt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)
    
    if opt.check_point!='/':
        ckpt = load_checkpoint(opt.check_point)
        load_param_into_net(net, ckpt)
    model = Model(net, net_loss, optt, metrics={"Accuracy": Accuracy()})
    print('-------------------starting training----------------------------')
    model.train(config.epoch_size, ds, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False)


if __name__ == '__main__':
    if opt.do_train=='1':
        ans()
    else:
        net = InceptionV4()
        lr = 0.005
        ds_test = create_dataset('./data/test', True, batch_size=config.batch_size)
        net_loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
        optt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                        config.weight_decay, config.loss_scale)

        ckpt = load_checkpoint(opt.check_point)
        load_param_into_net(net, ckpt)

        model = Model(net, net_loss, optt, metrics={"Accuracy": Accuracy()})

        acc = model.eval(ds_test)

        print('Accuracy of model is: ', acc)


