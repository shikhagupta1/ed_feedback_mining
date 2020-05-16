'''
@author Harshit

'''

from pyspark.sql import *
spark = SparkSession.builder \
    .master("local") \
    .appName("PISA") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

import re, csv, os, sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pyspark.sql import functions as F
from pyspark.sql.functions import mean, stddev
from pyspark.ml.feature import Imputer
import pandas as pd
from sklearn.model_selection import train_test_split



def preprocess(files):
    '''
    Read (lazy evaluation) all csv and create spark dataframe
    Parameters:
        - list of csvs
    Returns:
        - list of spark dfs
    '''
    dfs = []

    # iterate through all CSVs
    for file in files:
        df = spark.read.csv(
            file, inferSchema=True,
            header=True)
        
        # take out only ST* and PV* columns
        # as these have feed_back and perf data
        col_st = [c for c in df.columns if re.match('ST[0-9].*', c) and 'D' not in c]
        col_pv = [c for c in df.columns if re.match('PV[0-9].*', c)]
        df = df[col_st + col_pv]
        dfs.append(df)

    return dfs


def removeInvalidCols(df):
    '''
    Remove columns with all Nans or all same values
    Parameter:
        - dataframe
    Returns:
        - dataframe (after removing cols)
    '''
    count_distinct = df.select([
        F.approx_count_distinct(c).alias("{0}".format(c))
        for c in df.columns
    ])

    # consider only columns with >= 2 distincts
    distinct = count_distinct.toPandas().to_dict(orient='list')
    to_consider = [k for k in distinct if distinct[k][0] >= 2]
    
    # return the cleaned dataframe
    return df[to_consider]


def fillMedian(df):
    '''
    Fill cells with all Nans as Median
    Parameter:
        - dataframe
    Returns:
        - dataframe (after removing cols)
    '''
    cols = df.columns
    imputer = Imputer(
        strategy='median',
        inputCols=cols,
        # outputCols=['{}_clean'.format(c) for c in cols]
        outputCols=cols
    )

    df = imputer.fit(df).transform(df)
    # newCols = ['{}_clean'.format(c) for c in cols]
    # df = df[newCols]
    # df = df.rename(
    #     columns={
    #         name: name.split('_')[0] for name in newCols
    #     }
    # )
    return df


def fillMean(df):
    '''
    Fill cells with all Nans as mean
    Parameter:
        - dataframe
    Returns:
        - dataframe (after removing cols)
    '''
    cols = df.columns
    imputer = Imputer(
        strategy='median',
        inputCols=cols,
        # outputCols=['{}_clean'.format(c) for c in cols]
        outputCols=cols
    )

    df = imputer.fit(df).transform(df)
    # newCols = ['{}_clean'.format(c) for c in cols]
    # df = df[newCols]
    # df = df.rename(
    #     columns={
    #         name: name.split('_')[0] for name in newCols
    #     },
    # )
    return df


def standardize(df):
    '''
    Standardize all columns
    formula = [(x - mean) / std]
    CREDITS: https://gist.github.com/morganmcg1/15a9de711b9c5e8e1bd142b4be80252d
    Based on the solution on stackoverflow: 
    https://stackoverflow.com/questions/44580644/subtract-mean-from-pyspark-dataframe
    Paramters:
        - dataframe
    Returns:
        - dataframe with standardized variables
    '''

    # create list to aggregate all mean and stds
    aggAvg = []
    aggStd = []

    # aggregate all means and stds
    for c in df.columns:
        aggAvg.append(mean(df[c]).alias(c))
        aggStd.append(stddev(df[c]).alias(c + '_stddev'))
    
    averages = df.agg(*aggAvg).collect()[0]
    std_devs = df.agg(*aggStd).collect()[0]

    for c in df.columns:            
        df = df.withColumn(
            c + '_norm',
            ((df[c] - averages[c]) / std_devs[c + '_stddev'])
        )
    
    return df


def mergeAll(dfs, feat, tar):
    '''
    1. Merge all year data provided with df for each year
    2. Remove Nan and clean
    3. Standardize
    Parameters:
        - List of dfs
        - features dir
        - target dir
    Returns:
        - df (merged,clean,standardized)
    '''

    df = dfs[0]
    # take only ST and PV columns
    col_st = [
        c for c in df.columns if re.match('ST[0-9].*', c) and 'D' not in c]
    col_pv = [
        c for c in df.columns if re.match('PV[0-9].*', c)]
    for other_df in dfs[1:]:

        # due to difference in each year's columns
        # first take ST and PV columns
        other_col_st = [
            c for c in other_df.columns if re.match('ST[0-9].*', c) and 'D' not in c]
        other_col_pv = [
            c for c in other_df.columns if re.match('PV[0-9].*', c)]

        # take set intersection between above
        col_st = list(set(col_st) & set(other_col_st))
        col_pv = list(set(col_pv) & set(other_col_pv))
        df = df[col_st + col_pv].union(other_df[col_st + col_pv])
    
    # data cleaning
    X, Y = df[col_st], df[col_pv]

    # count distinct values in each columns
    X, Y = removeInvalidCols(X), removeInvalidCols(Y)
    
    print('Invalid cols removed')
    # take median of feedback info
    X = fillMedian(X)
    # take mean of perf info
    Y = fillMean(Y)

    print('NAN filled with mean/median')
    # standardize all variables
    standardize(X)
    standardize(Y)
    print('Data Standardized')

    num_partitions = X.rdd.getNumPartitions()
    # X.write.csv(feat)
    # Y.write.csv(tar)
    # print(X.rdd.getNumPartitions(), Y.rdd.getNumPartitions())
    X.toPandas().to_csv(feat+'.csv', index=False)
    Y.toPandas().to_csv(tar+'.csv', index=False)
    return num_partitions


def train_and_eval(num_files, feat, tar):
    '''
    Load data for each partition and train
    Parameters:
        - num_files
        - features dir
        - target dir
    '''

    # create partition and csv mappings
    feat_dict = {int(f.split('-')[1]):f for f in os.listdir(feat) if '.csv' in f}
    tar_dict = {int(f.split('-')[1]):f for f in os.listdir(tar) if '.csv' in f}
    
    # use mirrored strategy to evaluate/train
    # All Reduce Synchronous
    # different gpu, 1 CPU
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # use Multi Worker Mirrored strategy
    # currently experimental
    # All Reduce Synchronous
    # different nodes -> each node different gpus
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # use first 0.8 for train and last 0.2 for eval
    # concat all train csvs
    print('[START] Loading train features')
    x_train = pd.concat([
        pd.read_csv(
            os.path.join(feat, feat_dict[i]), header=0,
            encoding='ISO-8859-1')
        for i in range(int(0.8*num_files))
    ])
    print('[END] Loading train features')

    # get all target csvs
    print('[START] Loading train targets')
    y_train = pd.concat([
        pd.read_csv(
            os.path.join(feat, tar_dict[i]), header=0,
            encoding='ISO-8859-1')
        for i in range(int(0.8*num_files))
    ])
    print('[END] Loading train targets')

    # get eval data
    x_val = pd.concat([
        pd.read_csv(
            os.path.join(feat, feat_dict[i]), header=0,
            encoding='ISO-8859-1')
        for i in range(int(0.8*num_files), num_files)
    ])

    y_val = pd.concat([
        pd.read_csv(
            os.path.join(feat, tar_dict[i]), header=0,
            encoding='ISO-8859-1')
        for i in range(int(0.8*num_files), num_files)
    ])

    train = tf.data.Dataset.from_tensor_slices(
        (x_train.values, y_train.values)
    )

    val = tf.data.Dataset.from_tensor_slices(
        (x_val.values, y_val.values)
    )

    # set BATCH PARAMETERS
    BUFFER_SIZE = 1000

    BATCH_SIZE_PER_REPLICA = 100
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val = val.batch(BATCH_SIZE)


    # create model / network within strategy scope, to replicate to all nodes/gpu
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, input_shape=(219,), activation='relu'),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
        ])

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction='auto'),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ]
        )
        model.summary()

    # add callbacks
    # scheckpoints
    checkpoint_dir = './ckpt'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # decay function
    def decay(epoch):
        if epoch < 3:   return 0.001
        elif epoch < 7: return 0.0003
        else:   return 0.0001

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        ),
        tf.keras.callbacks.LearningRateScheduler(decay),
    ]

    # train the model and show results
    model.fit(
        train, epochs=15,
        callbacks=callbacks,
        validation_data=val
    )


def train_and_eval(feat, tar):
    '''
    Load data for each partition and train
    Parameters:
        - num_files
        - features dir
        - target dir
    '''

    
    # use mirrored strategy to evaluate/train
    # All Reduce Synchronous
    # different gpu, 1 CPU
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # use Multi Worker Mirrored strategy
    # currently experimental
    # All Reduce Synchronous
    # different nodes -> each node different gpus
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # use first 0.8 for train and last 0.2 for eval
    # concat all features
    print('[START] Loading features')
    x = pd.read_csv(feat)
    print('[END] Loading train features')

    # get all target
    print('[START] Loading targets')
    y = pd.read_csv(tar)
    print('[END] Loading targets')

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

    # convert to tensors
    train = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
    val = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))

    # set BATCH PARAMETERS
    BUFFER_SIZE = 1000

    BATCH_SIZE_PER_REPLICA = 100
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    train = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    val = val.batch(BATCH_SIZE)


    # create model / network within strategy scope, to replicate to all nodes/gpu
    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                1024, input_shape=(len(x.columns),), activation='relu'
            ),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(y.columns), activation='relu'),
        ])

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(reduction='auto'),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mae')
            ]
        )
        model.summary()

    # add callbacks
    # scheckpoints
    checkpoint_dir = './ckpt'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    # decay function
    def decay(epoch):
        if epoch < 3:   return 0.001
        elif epoch < 7: return 0.0003
        else:   return 0.0001

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        ),
        tf.keras.callbacks.LearningRateScheduler(decay),
    ]

    # train the model and show results
    model.fit(
        train, epochs=15,
        callbacks=callbacks,
        validation_data=val
    )



if __name__ == "__main__":
    files = ['cy6_ms_cmb_stu_qqq.csv', 'cy07_msu_stu_qqq.csv']

    print('[START] Preprocessing files')
    # dfs = preprocess(files)
    print('[END] Preprocessing files')

    feat, tar = 'features', 'target'
    print('[START] Merging dfs')
    # num_files = mergeAll(dfs, feat, tar)
    print('[END] Merging dfs')

    num_files=31
    print('[START] Tensorflow')
    # train_and_eval(num_files, feat, tar)
    train_and_eval(feat+'.csv', tar+'.csv')
    print('[END] Tensorflow')
