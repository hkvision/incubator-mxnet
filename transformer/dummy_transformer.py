import random
import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

batch_size = 32000
num_epoch = 10
model_prefix = 'drivethru_attention_d'
n_plus, n_time, n_bkids, n_weather, n_feels = 522, 167, 126, 35, 20
total = 400000

records = []

for i in range(0, total):
    pluids = [random.randint(0, n_plus - 1) for i in range(0, 5)]
    timeidx = random.randint(0, n_time - 1)
    bkidx = random.randint(0, n_bkids - 1)
    weatheridx = random.randint(0, n_weather - 1)
    feelsBucket = random.randint(0, n_feels - 1)
    label = random.randint(0, 1)
    records.append((pluids, timeidx, bkidx, weatheridx, feelsBucket, label))

data = pd.DataFrame(records,
                    columns=['pluids', 'timeidx', 'bkidx', 'weatheridx', 'feelsBucket', 'label'])


train, test = train_test_split(data, test_size=0.1, random_state=100)


X_train = mx.io.NDArrayIter(data={'pluids': np.array(train['pluids'].values.tolist(), dtype=int),
                                  'bkidx': train['bkidx'].values,
                                  'timeidx': train['timeidx'].values,
                                  'feels_bucket': train['feelsBucket'].values,
                                  'weatheridx': train['weatheridx'].values},
                            label={'output_label': train['label'].values},
                            batch_size=batch_size,
                            shuffle=True)
X_eval = mx.io.NDArrayIter(data={'pluids': np.array(test['pluids'].values.tolist(), dtype=int),
                                 'bkidx': test['bkidx'].values,
                                 'timeidx': test['timeidx'].values,
                                 'feels_bucket': test['feelsBucket'].values,
                                 'weatheridx': test['weatheridx'].values},
                            label={'output_label': test['label'].values},
                            batch_size=batch_size,
                            shuffle=True)
y_true = mx.symbol.Variable('output_label')


pluids = mx.symbol.Variable('pluids')
bkidx = mx.symbol.Variable('bkidx')
timeidx = mx.symbol.Variable('timeidx')
feels_bucket = mx.symbol.Variable('feels_bucket')
weatheridx = mx.symbol.Variable('weatheridx')
plu_embed = mx.symbol.Embedding(data=pluids, input_dim=n_plus, output_dim=100, name='plu_embed')
bkidx_embed = mx.symbol.Embedding(data=bkidx, input_dim=n_bkids, output_dim=200, name='bkid_embed')
time_embed = mx.symbol.Embedding(data=timeidx, input_dim=n_time, output_dim=200, name='time_embed')
feels_embed = mx.symbol.Embedding(data=feels_bucket, input_dim=n_feels, output_dim=200, name='feels_embed')
weather_embed = mx.symbol.Embedding(data=weatheridx, input_dim=n_weather, output_dim=200, name='weather_embed')

from gluonnlp.model.transformer import TransformerEncoder
encoder = TransformerEncoder(units=100, hidden_size=256, num_heads=4, num_layers=2, max_length=5, dropout=0.1)
encoder_output, att = encoder.hybrid_forward(mx.sym, inputs=plu_embed)
flatten = mx.symbol.flatten(encoder_output, "flatten")
encoder_features = mx.symbol.FullyConnected(data=flatten, num_hidden=100, name='encoder_features')

context_features = mx.symbol.broadcast_mul((1 + bkidx_embed + time_embed + weather_embed + feels_embed),
                                           encoder_features, name='latent_cross')
ac1 = mx.symbol.Activation(data=context_features, act_type="relu", name="relu1")
dropout1 = mx.symbol.Dropout(data=ac1, p=0.3, name="dropout1")
fc1 = mx.symbol.FullyConnected(data=dropout1, num_hidden=int(n_plus), name='fc1')
rec_model = mx.symbol.SoftmaxOutput(data=fc1, label=y_true, name='output')

mod = mx.mod.Module(symbol=rec_model,
                    data_names=['pluids', 'bkidx', 'timeidx', 'feels_bucket', 'weatheridx'],
                    label_names=['output_label'],
                    context=[mx.cpu()])
from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')
profiler.set_state('run')
mod.fit(train_data=X_train,
        num_epoch=num_epoch,
        initializer=mx.init.Xavier(rnd_type="gaussian"),
        optimizer='adagrad',
        eval_metric=['accuracy'],
        validation_metric=['accuracy', mx.metric.TopKAccuracy(3)],
        eval_data=X_eval,
        batch_end_callback=mx.callback.Speedometer(batch_size, 2))
profiler.set_state('stop')
# Dump all results to log file before download
profiler.dump()
