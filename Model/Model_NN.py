import numpy as np
import pandas as pd
import gc
import random
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder

from keras.layers import Dense, Input, Dropout, ReLU, LeakyReLU
from keras.layers import Add
from keras.layers import Embedding, concatenate, Reshape
from keras.optimizers import RMSprop, Adam
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint


def train_mix_generate_full(df_input, sc):

    df_use = df_input.copy()
    Train_df = df_use[df_use['date_block_num'] != df_use['date_block_num'].max()]
    Test_df = df_use[df_use['date_block_num'] == df_use['date_block_num'].max()]
    
    Train_df[feature_norm] = sc.fit_transform(Train_df[feature_norm])
    Test_df[feature_norm] = sc.transform(Test_df[feature_norm])
    
    Train_df['shop_id'] = Train_df['shop_id'] - shop_min
    Train_df['item_category_id'] = Train_df['item_category_id'] - item_cat_min
    Train_df['type_code'] = Train_df['type_code'] - type_code_min
    Train_df['subtype_code'] = Train_df['subtype_code'] - subtype_code_min
    Train_df['city_code'] = Train_df['city_code'] - city_code_min
    Train_df['month'] = Train_df['month'] - month_min

    Test_df['shop_id'] = Test_df['shop_id'] - shop_min
    Test_df['item_category_id'] = Test_df['item_category_id'] - item_cat_min
    Test_df['type_code'] = Test_df['type_code'] - type_code_min
    Test_df['subtype_code'] = Test_df['subtype_code'] - subtype_code_min
    Test_df['city_code'] = Test_df['city_code'] - city_code_min
    Test_df['month'] = Test_df['month'] - month_min
    
    X_train = Train_df.loc[Train_df.date_block_num < 33][features]
    Y_train = Train_df.loc[Train_df.date_block_num < 33]['item_cnt_month']
    X_valid = Train_df.loc[Train_df.date_block_num == 33][features]
    Y_valid = Train_df.loc[Train_df.date_block_num == 33]['item_cnt_month']
    X_test = Test_df[features]
    
    del Train_df, Test_df
    gc.collect()
    
    return X_train, Y_train, X_valid, Y_valid, X_test, df


def NN_layers(initializer, X_train, shop_NN, item_cat_NN, type_NN, subtype_NN, city_NN, month_NN,
              n_features = 30, dense_features = 220, drop_out = 0.05, embed_dropout = 0.1):
    
    shop_input = Input(shape = (1,))
    item_cat_input = Input(shape = (1,))
    type_input = Input(shape = (1,))
    subtype_input = Input(shape = (1,))
    city_input = Input(shape = (1,))
    month_input = Input(shape = (1,))
    
    layer_input = Input(shape = (X_train.shape[1] - 6,))
    
    embedding_shop = Embedding(shop_NN.max() + 1, n_features)
    embedding_item_cat = Embedding(item_cat_NN.max() + 1, n_features)
    embedding_type = Embedding(type_NN.max() + 1, n_features)
    embedding_subtype = Embedding(subtype_NN.max() + 1, n_features)
    embedding_city = Embedding(city_NN.max() + 1, n_features)
    embedding_month = Embedding(month_NN.max() + 1, n_features)
    
    shop = Reshape((-1,))(embedding_shop(shop_input))
    item_cat = Reshape((-1,))(embedding_item_cat(item_cat_input))
    Type = Reshape((-1,))(embedding_type(type_input))
    subtype = Reshape((-1,))(embedding_subtype(subtype_input))
    city = Reshape((-1,))(embedding_city(city_input))
    month = Reshape((-1,))(embedding_month(month_input))
    
    x = concatenate(
        [shop, item_cat, Type, subtype, city, month, layer_input], axis = -1)
    
    x = Dense(dense_features, input_dim=X_train.shape[1], kernel_initializer=initializer, activation='relu')(x)
    
    x = Dense(dense_features, kernel_initializer=initializer)(x)
    x = ReLU(negative_slope=0.)(x)
    x = Add()([Dense(dense_features, kernel_initializer=initializer)(x), x])
    x = ReLU(negative_slope=0.)(x)
    x = Dropout(drop_out)(x)
    x = Dense(dense_features, kernel_initializer=initializer)(x)
    x = ReLU(negative_slope=0.)(x)
    x = Add()([Dense(dense_features, kernel_initializer=initializer)(x), x])
    x = ReLU(negative_slope=0.)(x)
    x = Dropout(drop_out)(x)
    x = Dense(dense_features, kernel_initializer=initializer)(x)
    x = ReLU(negative_slope=0.)(x)
    x = Add()([Dense(dense_features, kernel_initializer=initializer)(x), x])
    x = ReLU(negative_slope=0.)(x)
    x = Dropout(drop_out)(x)
    x = Dense(dense_features, kernel_initializer=initializer)(x)
    x = ReLU(negative_slope=0.)(x)
    x = Add()([Dense(dense_features, kernel_initializer=initializer)(x), x])
    x = ReLU(negative_slope=0.)(x)
    x = Dropout(drop_out)(x)
    x = Dense(dense_features, kernel_initializer=initializer)(x) 
    x = ReLU(negative_slope=0.)(x)
    x = Dense(dense_features, kernel_initializer=initializer)(x)
    x = ReLU(negative_slope=0.)(x)
    x = Dropout(drop_out)(x)

    x = Dense(1, kernel_initializer='uniform')(x)
    
    return shop_input, item_cat_input, type_input, subtype_input, city_input, month_input, layer_input, x


class train_config:
    
    def __init__(self, features, seed):
        
        self.features = features
        self.seed = seed


def model_train(train_mix, train_config):
    
    X_train, Y_train, X_valid, Y_valid, X_test, train_mix_old = train_mix_generate_full(train_mix)
    feature_norm = list(set(train_config.features).difference(
        ['shop_id', 'item_category_id', 'type_code', 'subtype_code', 'city_code', 'month', 'date_block_num'])
                       )
    
    shop_NN = train_mix['shop_id'].unique() - train_mix['shop_id'].min()
    item_cat_NN = train_mix['item_category_id'].unique() - train_mix['item_category_id'].min()
    type_NN = train_mix['type_code'].unique() - train_mix['type_code'].min()
    subtype_NN = train_mix['subtype_code'].unique() - train_mix['subtype_code'].min()
    city_NN = train_mix['city_code'].unique() - train_mix['city_code'].min()
    month_NN = train_mix['month'].unique() - train_mix['month'].min()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    initializer = keras.initializers.RandomUniform(seed=train_config.seed)
    # Should have better way for model parameters management
    shop_input, item_cat_input, type_input, subtype_input, city_input, month_input, layer_input, layer_output =     NN_layers(initializer, X_train, shop_NN, item_cat_NN, type_NN, subtype_NN, city_NN, month_NN, 30, 100, 0.2, 0.1)
    model_NN = Model([shop_input, item_cat_input, type_input, subtype_input, city_input, month_input, layer_input], layer_output)
    
    lr = 0.00005
    beta_1 = 0.9
    beta_2 = 0.999
    model_NN.compile(loss='mse', optimizer=Adam(lr = lr, beta_1 = beta_1, beta_2 = beta_2), metrics=['mse'])
    checkpointer = ModelCheckpoint(filepath=f'../NN_models/NN_{train_config.seed}.hdf5', verbose=1, save_best_only=True)
    
    Batch_size =  128
    Num_Epoch = 10
    model_NN.fit([X_train['shop_id'],
                  X_train['item_category_id'],
                  X_train['type_code'],
                  X_train['subtype_code'],
                  X_train['city_code'],
                  X_train['month'],
                  X_train[feature_norm]
                 ],
                 Y_train,
                 batch_size = Batch_size,
                 epochs = Num_Epoch,
                 validation_data=([X_valid['shop_id'],
                                    X_valid['item_category_id'],
                                    X_valid['type_code'],
                                    X_valid['subtype_code'],
                                    X_valid['city_code'],
                                    X_valid['month'],
                                    X_valid[feature_norm]
                                   ], 
                                   Y_valid),
                  verbose = 2,
                 callbacks = [checkpointer]
             )
    
    model_NN.load_weights(f'../NN_models/NN_{train_config.seed}.hdf5')
    
    return model_NN


if __name__ == "__main__":
    
    with open('../processed/train_mix.pickle', 'rb') as handle:
        train_mix = pickle.load(handle)
                       
    config_path = sys.argv[1]
    with open(config_path) as json_file:
        config = json.load(json_file)
        
    t_config = train_config(
        features = config['features'],
        seed = config['seed']
    )
    
    model_NN = model_train(train_mix, t_config)
    
    Y_test_NN = model_NN.predict([X_test['shop_id'],
                              X_test['item_category_id'],
                              X_test['type_code'],
                              X_test['subtype_code'],
                              X_test['city_code'],
                              X_test['month'],
                              X_test[feature_norm]
             ])
    
    sample_submission = pd.read_csv('../input/sample_submission.csv')
    sample_submission['item_cnt_month'] = Y_test_NN
    sample_submission.to_csv(f'../results/w_history/submission_w_history_NN_{t_config.seed}.csv', index=False)
