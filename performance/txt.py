import logging
import time
import random

import mxnet as mx
import numpy as np
import pandas as pd
from gluonnlp.model.transformer import TransformerEncoder, TransformerEncoderCell
from mxnet import gluon
from sklearn.model_selection import train_test_split


batch_size = 1000
num_epoch = 4
ctx = mx.cpu()
context_cols = ['bkidx', 'timeidx', 'modeidx', 'deviceidx', 'carrieridx']
sequence_col = "pluids"
vl_col = "valid_len"

# Setting log level
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(filename='./gluon_0720', level=logging.DEBUG)


class MeanMaxPooling(gluon.nn.HybridBlock):
    def __init__(self, axis=1, dropout=0.0, prefix=None, params=None, **kwargs):
        super(MeanMaxPooling, self).__init__(**kwargs)
        #         super().__init__(prefix=prefix, params=params)
        self.axis = axis
        self.dropout = dropout

    def hybrid_forward(self, F, inputs):
        mean_out = F.mean(data=inputs, axis=self.axis)
        max_out = F.max(data=inputs, axis=self.axis)
        outputs = F.concat(mean_out, max_out, dim=1)
        if self.dropout:
            outputs = F.Dropout(data=outputs, p=self.dropout)
        #         outputs = F.LayerNorm(outputs)
        return outputs


class SequenceTransformer(gluon.nn.HybridBlock):
    def __init__(self, num_items, item_embed, item_hidden_size, item_max_length, item_num_heads,
                 item_num_layers, item_transformer_dropout, item_pooling_dropout, cross_size,
                 prefix=None, params=None, **kwargs):
        super(SequenceTransformer, self).__init__(**kwargs)
        #         super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.item_pooling_dp = MeanMaxPooling(dropout=item_pooling_dropout)
            self.item_encoder = TransformerEncoder(units=item_embed,
                                                   hidden_size=item_hidden_size,
                                                   num_heads=item_num_heads,
                                                   num_layers=item_num_layers,
                                                   max_length=item_max_length,
                                                   dropout=item_transformer_dropout)
            self.embedding = gluon.nn.Embedding(input_dim=num_items, output_dim=item_embed)
            self.dense = gluon.nn.Dense(cross_size)

    def hybrid_forward(self, F, input_item, item_valid_length=None):
        item_embed_out = self.embedding(input_item)
        item_encoding, item_att = self.item_encoder(inputs=item_embed_out, valid_length=item_valid_length)
        item_out = self.item_pooling_dp(item_encoding)
        item_out = self.dense(item_out)

        return item_out


class ContextTransformer(gluon.nn.HybridBlock):
    def __init__(self, context_dims, context_embed, context_hidden_size,
                 context_num_heads, context_transformer_dropout, context_pooling_dropout,
                 cross_size, prefix=None, params=None, **kwargs):
        super(ContextTransformer, self).__init__(**kwargs)
        #         super().__init__(prefix=prefix, params=params)
        self.context_dims = context_dims
        self.context_embed = context_embed
        self.cross_size = cross_size
        with self.name_scope():
            self.context_pooling_dp = MeanMaxPooling(dropout=context_pooling_dropout)
            self.context_encoder = TransformerEncoderCell(units=context_embed,
                                                          hidden_size=context_hidden_size,
                                                          num_heads=context_num_heads,
                                                          dropout=context_transformer_dropout
                                                          )
            self.dense = gluon.nn.Dense(self.cross_size)
            self.embeddings = gluon.nn.HybridSequential()
            for i, context_dim in enumerate(self.context_dims):
                self.embeddings.add(gluon.nn.Embedding(self.context_dims[i], self.context_embed))

    def hybrid_forward(self, F, input_context_list):
        context_embed = [self.embeddings[i](input_context) for i, input_context in enumerate(input_context_list)]
        context_input = []
        for i in context_embed:
            context_input.append(F.expand_dims(i, axis=1))
        context_embedding = F.concat(*context_input, dim=1)
        context_encoding, context_att = self.context_encoder(context_embedding)
        context_out = self.context_pooling_dp(context_encoding)
        context_out = self.dense(context_out)

        return context_out


class TxT(gluon.nn.HybridBlock):
    def __init__(self, num_items, context_dims, item_embed=100, context_embed=100,
                 item_hidden_size=256, item_max_length=8, item_num_heads=4, item_num_layers=2,
                 item_transformer_dropout=0.0, item_pooling_dropout=0.1, context_hidden_size=256,
                 context_num_heads=2, context_transformer_dropout=0.0, context_pooling_dropout=0.0,
                 act_type="gelu", cross_size=100, prefix=None, params=None, **kwargs):
        super(TxT, self).__init__(**kwargs)
        self.act_type = act_type
        with self.name_scope():
            self.sequence_transformer = SequenceTransformer(
                num_items=num_items,
                item_embed=item_embed,
                item_hidden_size=item_hidden_size,
                item_max_length=item_max_length,
                item_num_heads=item_num_heads,
                item_num_layers=item_num_layers,
                item_transformer_dropout=item_transformer_dropout,
                item_pooling_dropout=item_pooling_dropout,
                cross_size=cross_size,
                prefix=prefix, params=params
            )
            self.context_transformer = ContextTransformer(
                context_dims=context_dims,
                context_embed=context_embed,
                context_hidden_size=context_hidden_size,
                context_num_heads=context_num_heads,
                context_transformer_dropout=context_transformer_dropout,
                context_pooling_dropout=context_pooling_dropout,
                cross_size=cross_size,
                prefix=prefix, params=params
            )
            self.dense1 = gluon.nn.Dense(units=num_items//2)
            if act_type == "relu":
                self.act = gluon.nn.Activation(activation="relu")
            elif act_type == "gelu":
                self.act = gluon.nn.GELU()
            elif act_type == "leakyRelu":
                self.act = gluon.nn.LeakyReLU(alpha=0.2)
            else:
                raise NotImplementedError
            self.dense2 = gluon.nn.Dense(units=num_items, activation=None)

    def hybrid_forward(self, F, input_item, item_valid_length, input_context_list):
        item_outs = self.sequence_transformer(input_item, item_valid_length)
        context_outs = self.context_transformer(input_context_list)

        outs = F.broadcast_mul(item_outs, context_outs)
        outs = self.dense1(outs)
        outs = self.act(outs)
        outs = self.dense2(outs)

        return outs


n_plus, n_time, n_bkids, n_mode, n_brand, n_carrier = 681, 167, 6070, 4, 150, 420
logger.info([n_plus, n_time, n_bkids, n_mode, n_brand, n_carrier])


def valid_len(row):
    seq = row['pluids']
    seq = [p for p in seq if p != 0]
    vl = len(seq)

    return vl


total = 40000
records = []

for i in range(0, total):
    pluids = [float(random.randint(1, n_plus - 1)) for i in range(0, 3)] + [0.0] * 5
    timeidx = float(random.randint(0, n_time - 1))
    bkidx = float(random.randint(0, n_bkids - 1))
    modeidx = float(random.randint(0, n_mode - 1))
    deviceidx = float(random.randint(0, n_brand - 1))
    carrieridx = float(random.randint(0, n_carrier - 1))
    label = float(random.randint(1, n_plus - 1))
    records.append((pluids, timeidx, bkidx, modeidx, deviceidx, carrieridx, label))

data = pd.DataFrame(records,
                    columns=['pluids', 'timeidx', 'bkidx', 'modeidx', 'deviceidx', 'carrieridx', 'label'])
data[vl_col] = data.apply(valid_len, axis=1)
logger.debug(data)
print(data)

train, test = train_test_split(data, shuffle=False, train_size=0.9, random_state=100)

train_data = {col: train[col].values for col in context_cols}
train_data[vl_col] = train[vl_col].values
train_data[sequence_col] = np.array(train[sequence_col].values.tolist(), dtype=int)
test_data = {col: test[col].values for col in context_cols}
test_data[vl_col] = test[vl_col].values
test_data[sequence_col] = np.array(test[sequence_col].values.tolist(), dtype=int)

X_train = mx.io.NDArrayIter(data=train_data,
                            label={'output_label': train['label'].values},
                            batch_size=batch_size,
                            shuffle=True)
X_eval = mx.io.NDArrayIter(data=test_data,
                           label={'output_label': test['label'].values},
                           batch_size=batch_size,
                           shuffle=True)
y_true = mx.symbol.Variable('output_label')


class DataIterLoader:
    def __init__(self, data_iter, seq_col, vl_col, context_cols):
        self.data_iter = data_iter
        self.seq_col = seq_col
        self.vl_col = vl_col
        self.context_cols = context_cols

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        # assert len(batch.data) == len(batch.label)
        data = batch.data
        label = batch.label[0]
        desc_list = list(x[0] for x in self.data_iter.provide_data)
        sequence_idx = desc_list.index(self.seq_col)
        vl_idx = desc_list.index(self.vl_col)
        context_idx = [desc_list.index(c) for c in self.context_cols]
        sequence_data = data[sequence_idx]
        valid_length = data[vl_idx]
        context_data = [data[i] for i in context_idx]
        return (sequence_data, valid_length, context_data), label

    def next(self):
        return self.__next__()


train_dataloader = DataIterLoader(X_train, sequence_col, vl_col, context_cols)
test_dataloader = DataIterLoader(X_eval, sequence_col, vl_col, context_cols)

for (sequence_data, valid_len, context_data), label in train_dataloader:
    print(sequence_data)
    break

logger.info("data has been loaded to ndArrayIter")

txt = TxT(
    num_items=n_plus,
    context_dims=[n_bkids, n_time, n_mode, n_brand, n_carrier]
)


txt.hybridize(static_alloc=True, static_shape=True)
txt.initialize(mx.init.Xavier(rnd_type="gaussian"), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(
    txt.collect_params(),
    'adam',
    {'learning_rate': 0.001}
)


loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
acc = mx.metric.Accuracy()
top2 = mx.metric.TopKAccuracy(2)
mini_metric = mx.metric.Accuracy()
mini_metric2 = mx.metric.TopKAccuracy(5)
log_interval = 5


def evaluate_accuracy(data_iterator, txt):
    acc = mx.metric.Accuracy()
    for i, ((sequence_data, valid_length, context_data), label) in enumerate(data_iterator):
        sequence = sequence_data.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        context = [d.as_in_context(ctx) for d in context_data]
        label = label.as_in_context(ctx)
        output = txt(sequence, valid_length, context)
        predictions = mx.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


for epoch in range(num_epoch):
    # start of epoch
    b = 0
    top2.reset()
    # Epoch training stats
    start_epoch_time = time.time()
    epoch_L = 0.0
    epoch_sent_num = 0
    epoch_wc = 0
    # Log interval training stats
    start_log_interval_time = time.time()
    log_interval_wc = 0
    log_interval_sent_num = 0
    step_loss = 0.0

    for i, ((sequence_data, valid_length, context_data), label) in enumerate(train_dataloader):
        # Load the data to the ctx
        sequence = sequence_data.as_in_context(ctx)
        valid_length = valid_length.as_in_context(ctx)
        context = [d.as_in_context(ctx) for d in context_data]
        label = label.as_in_context(ctx)

        with mx.autograd.record():
            # Forward computation
            output = txt(sequence, valid_length, context)
            loss = loss_fn(output, label)
            predictions = mx.nd.argmax(output, axis=1)

        # Backward compution
        loss.backward()

        # Update parameter
        trainer.step(batch_size)
        step_loss += loss.mean().asscalar()
        acc.update(predictions, label)

        if (i + 1) % log_interval == 0:
            log_time = time.time() - start_log_interval_time
            print(
                '[Epoch {} Batch {}] elapsed {:.2f} s, '
                'avg loss={:.6f}, lr={:.6f}, metric={:.6f}, throughput={:.6f}'.format(
                    epoch,
                    i + 1,
                    log_time,
                    step_loss / log_interval,
                    trainer.learning_rate,
                    float(acc.get()[1]),
                    batch_size * log_interval / log_time,
                )
            )
            # Clear log interval training stats
            start_log_interval_time = time.time()
            log_interval_wc = 0
            log_interval_sent_num = 0
            step_loss = 0
            acc.reset()

    test_accuracy = evaluate_accuracy(train_dataloader, txt)
    train_accuracy = evaluate_accuracy(test_dataloader, txt)
    print("Epoch {}. Train_acc {:.6f}, Test_acc {:.6f}".format(epoch, train_accuracy, test_accuracy))

