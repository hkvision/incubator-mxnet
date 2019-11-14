import ray.services
from contextlib import closing
import socket
from dmlc_tracker.tracker import get_host_ip
from dmlc_tracker.ssh import get_env
import subprocess
from threading import Thread
import multiprocessing
import mxnet as mx


class MXNetRunner(object):
    """Manages a MXNet model for training."""

    def __init__(self, model_creator, data_creator, args):
        self.model_creator = model_creator
        self.data_creator = data_creator
        self.args = args
        self.epoch = 0

    def setup_distributed(self, env, prog):
        env["DMLC_NODE_HOST"] = self.get_node_ip()
        self.env = env
        # def run(prog):
        #     subprocess.check_call(prog, shell = True)
        #
        # from threading import Thread
        # prog = prog + " python create_kv.py"
        # thread = Thread(target=run, args=(prog, ))
        # thread.setDaemon(True)
        # thread.start()

        # import os
        # os.environ = env
        # print(os.environ["DMLC_NUM_WORKER"])
        # kv = mx.kv.create(self.args.kvstore)
        # print(kv.num_workers)
        # q = multiprocessing.Queue()
        # self.train_data, self.val_data = self.data_creator(self.args, self.kv)
        # self.trainer, self.loss = self.model_creator(self.args, self.kv)
        # print(env)

        # proc1 = multiprocessing.Process(target=_create_kv, args=(self.env, self.args))
        # proc1.start()
        from threading import Thread
        thread = Thread(target=_create_kv, args=(self.env, self.args))
        thread.setDaemon(True)
        thread.start()

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
        self.num_runners = self.num_servers + self.num_workers # + 1  # One is for scheduler

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

        env = {'DMLC_NUM_WORKER': self.num_workers,
               'DMLC_NUM_SERVER': self.num_servers}
        hostIP = get_host_ip()
        port = 9001
        env['DMLC_PS_ROOT_URI'] = str(hostIP)
        env['DMLC_PS_ROOT_PORT'] = str(port)
        progs = []
        envs = []
        for i in range(self.num_workers + self.num_servers):
            current_env = env.copy()  # Need to copy other envs in os?
            current_env['DMLC_ROLE'] = 'server' if i < self.num_servers else 'worker'
            envs.append(current_env)
            prog = get_env(current_env)
            progs.append(prog)

        env['DMLC_ROLE'] = 'scheduler'
        # import os
        # for k, v in env.items():
        #     os.putenv(str(k), str(v))
        # kv = mx.kv.create(self.args.kvstore)
        # print(kv.num_workers)

        # proc1 = multiprocessing.Process(target=_create_kv, args=(env, self.args))
        # proc1.start()

        from threading import Thread
        thread = Thread(target=_create_kv, args=(env, self.args))
        thread.setDaemon(True)
        thread.start()

        # env['DMLC_ROLE'] = 'scheduler'
        # envs.append(env)
        # progs.append(get_env(env))
        # Get setup tasks in order to throw errors on failure
        ray.get([
            worker.setup_distributed.remote(envs[i], progs[i])
            for i, worker in enumerate(self.runners)
        ])

        def run(prog):
            subprocess.check_call(prog, shell = True)

        # envs.append(env)
        # prog = get_env(env)
        # prog = prog + " python create_kv.py"
        # thread = Thread(target=run, args=(prog, ))
        # thread.setDaemon(True)
        # thread.start()
        # apply envs to the driver


def _create_kv(env, args):
    import os
    # os.environ = env
    import mxnet as mx
    for k, v in env.items():
        os.putenv(str(k), str(v))
    import time
    # time.sleep(2)
    # print(os.environ["DMLC_ROLE"])
    # os.system("echo $DMLC_ROLE")
    os.system("echo $DMLC_NUM_WORKER")
    os.system("echo $DMLC_ROLE")
    # os.putenv("VARIABLE", "123")
    # os.system("echo $VARIABLE")
    kv = mx.kv.create(args.kvstore)
    print(kv.num_workers)
    print(kv.rank)
