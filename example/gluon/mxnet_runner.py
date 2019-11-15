import ray.services
from contextlib import closing
import socket
from dmlc_tracker.tracker import get_host_ip
import subprocess
import os


class MXNetRunner(object):
    """Manages a MXNet model for training."""

    def __init__(self, model_creator, data_creator, args):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.args = args
        self.is_worker = False
        self.epoch = 0

    def setup_distributed(self, env):
        env["DMLC_NODE_HOST"] = self.get_node_ip()
        self.env = env
        if env["DMLC_ROLE"] == "worker":
            self.is_worker = True

        if self.is_worker:
            os.environ.update(env)
            import mxnet as mx
            self.kv = mx.kv.create(self.args.kvstore)
            print(self.kv.num_workers)
            print(self.kv.rank)
        else:  # server or scheduler
            subprocess.Popen("python -c 'import mxnet'", shell=True, env=env)

    def step(self):
        """Runs a training epoch and updates the model parameters."""
        import time
        tic = time.time()
        self.train_data.reset()
        from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
        metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])
        metric.reset()
        btic = time.time()
        for i, batch in enumerate(self.train_data):
            from mxnet import gluon
            data = gluon.utils.split_and_load(batch.data[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0].astype("float32"), ctx_list=[mx.cpu()], batch_axis=0)
            outputs = []
            Ls = []
            from mxnet import autograd as ag
            with ag.record():
                for x, y in zip(data, label):
                    z = self.net(x)
                    L = self.loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                ag.backward(Ls)
            self.trainer.step(batch.data[0].shape[0])
            metric.update(label, outputs)
            if self.args.log_interval and not (i + 1) % self.opt.log_interval:
                name, acc = metric.get()
                import logging
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f, %s=%f' % (
                    self.epoch, i, self.args.batch_size / (time.time() - btic), name[0], acc[0], name[1], acc[1]))
            btic = time.time()

    def shutdown(self):
        """Attempts to shut down the worker."""
        # del self.model
        # del self.train_data
        # del self.val_data
        pass

    def get_node_ip(self):
        """Returns the IP address of the current node."""
        return ray.services.get_node_ip_address()

    def find_free_port(self):
        """Finds a free port on the current node."""
        return find_free_port()


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class MXNetTrainer(object):
    def __init__(self,
                 model_creator,
                 data_creator,
                 args):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.args = args
        self.num_workers = args.num_workers
        self.num_servers = args.num_servers if args.num_servers else self.num_workers
        self.num_runners = self.num_servers + self.num_workers

        # Generate actor class
        Runner = ray.remote(MXNetRunner)

        # Start workers
        self.runners = [
            Runner.remote(
                model_creator,
                data_creator,
                self.args)
            for i in range(self.num_runners)
        ]

        # Compute URL for initializing distributed setup
        ips = ray.get(
            [runner.get_node_ip.remote() for runner in self.runners])
        ports = ray.get(
            [runner.find_free_port.remote() for runner in self.runners])

        env = os.environ.copy()
        env.update({
            "DMLC_PS_ROOT_URI": str(get_host_ip()),
            "DMLC_PS_ROOT_PORT": str(find_free_port()),
            "DMLC_NUM_SERVER": str(self.num_servers),
            "DMLC_NUM_WORKER": str(self.num_workers),
        })
        envs = []
        for i in range(self.num_workers + self.num_servers):
            current_env = env.copy()
            current_env['DMLC_ROLE'] = 'server' if i < self.num_servers else 'worker'
            envs.append(current_env)

        env['DMLC_ROLE'] = 'scheduler'
        subprocess.Popen("python -c 'import mxnet'", shell=True, env=env)  # env need to contain system env to run bash

        ray.get([
            worker.setup_distributed.remote(envs[i])
            for i, worker in enumerate(self.runners)
        ])
