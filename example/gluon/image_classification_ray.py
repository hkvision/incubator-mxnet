# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import division

import argparse
import logging

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision as models

from data import get_cifar10_iterator

# logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('image-classification.log')
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)


def get_data_iters(args, kv):
    """get dataset iterators"""
    train_data, val_data = get_cifar10_iterator(args.batch_size, (3, 32, 32),
                                                num_parts=kv.num_workers, part_index=kv.rank)
    return train_data, val_data


def get_model(args):
    """Model initialization."""
    model = args.model
    context = [mx.cpu()]
    kwargs = {'ctx': context, 'pretrained': args.use_pretrained, 'classes': 10}
    if model.startswith('resnet'):
        kwargs['thumbnail'] = args.use_thumbnail
    elif model.startswith('vgg'):
        kwargs['batch_norm'] = args.batch_norm

    net = models.get_model(model, **kwargs)
    net.initialize(mx.init.Xavier(magnitude=2))
    net.cast("float32")
    net.collect_params().reset_ctx(context)
    return net


def get_loss(args):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    return loss


if __name__ == '__main__':
    # CLI
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('-n', '--num-workers', required=True, type=int,
                        help='number of worker nodes to be launched')
    parser.add_argument('-s', '--num-servers', type=int,
                        help='number of server nodes to be launched, \
                        in default it is equal to NUM_WORKERS')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--use_thumbnail', action='store_true',
                        help='use thumbnail or not in resnet. default is false.')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use. Default=123.')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    opt = parser.parse_args()

    import ray
    from example.gluon.mxnet_runner import MXNetTrainer
    ray.init()
    trainer = MXNetTrainer(get_data_iters, get_model, get_loss, opt)
    print("Training for one epoch started")
    train_stats = trainer.train()
    print("Training for one epoch ended")
    print("Training stats:")
    for stat in train_stats:
        print(stat)
    ray.shutdown()
