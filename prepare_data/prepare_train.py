import numpy as np
import pandas as pd
import re
from collections import Counter
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder

from process_shop import process_shop
from process_item import process_item
from process_item_cat import process_item_cat

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: #and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df


def merge_dataset(df):
    
    df = pd.merge(df, shops, on=['shop_id'], how='left')
    df = pd.merge(df, item, on=['item_id'], how='left')
    df = pd.merge(df, item_cat, on=['item_category_id'], how='left')
    df['city_code'] = df['city_code'].astype(np.int8)
    df['item_category_id'] = df['item_category_id'].astype(np.int8)
    df['type_code'] = df['type_code'].astype(np.int8)
    df['subtype_code'] = df['subtype_code'].astype(np.int8)
    
    return df


def target_generate(df):
    df = pd.merge(df, sales_group[['date_block_num','shop_id','item_id','revenue_month','item_cnt_month']], on=cols, how='left')
    df['item_cnt_month'] = (df['item_cnt_month']
                                    .fillna(0)
                                    .clip(0,20)
                                    .astype(np.float32))
    
    df['revenue_month'] = (df['revenue_month']
                                    .fillna(0)
                                    .astype(np.float32))
    return df


def lag_feature(df, lags = np.arange(12) + 1, groupcols = ['shop_id', 'item_id'], col = 'item_cnt_month', fillna = '0'):
    subset = ['date_block_num'] + groupcols
    new_col = '_'.join(groupcols)+'_avg_'+str(col)
    
    group = train_mix.groupby(subset).agg({col: ['mean']})
    group.columns = [new_col]
    group.reset_index(inplace=True)
    
    for i in lags:
        shifted = group.copy()
        shifted.columns = ['date_block_num'] + groupcols + [new_col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=(['date_block_num'] + groupcols), how='left')
        
        if fillna == 'median':
            df[f'{col}_lag_{i}'].fillna(df[new_col+'_lag_'+str(i)].median(), inplace = True)
        elif fillna == 'mean':
            df[f'{col}_lag_{i}'].fillna(df[new_col+'_lag_'+str(i)].mean(), inplace = True)
        elif fillna == '0':
            df.fillna(0, inplace = True)
        del shifted
        gc.collect()    
        
    del group
    
    return df


def FE_lag_statistic(train_mix, lags, groupcols, col):
    
    train_mix = lag_feature(train_mix,lags,groupcols,col,'no')
    new_col = '_'.join(groupcols)+'_avg_'+str(col)
    cols_new = [new_col+'_lag_'+str(i) for i in lags]
    cols_old = [new_col+'_lag_'+str(i) for i in [1,2,3]]
    
    train_mix.loc[train_mix['item_new']==1,'_'.join(groupcols)+'_max'] = train_mix[cols_new].max(axis=1)
    train_mix.loc[train_mix['item_new']==0,'_'.join(groupcols)+'_max'] = train_mix[cols_old].max(axis=1)
    train_mix['_'.join(groupcols)+'_max'] = train_mix['_'.join(groupcols)+'_max'].astype(np.float16)

    train_mix.drop(cols_new, axis=1, inplace = True)
    train_mix.fillna({'_'.join(groupcols)+'_max':0}, inplace = True)
    
    train_mix.loc[train_mix['item_new']==1,'_'.join(groupcols)+'_mean'] = train_mix[cols_new].mean(axis=1)
    train_mix.loc[train_mix['item_new']==0,'_'.join(groupcols)+'_mean'] = train_mix[cols_old].mean(axis=1)
    train_mix['_'.join(groupcols)+'_mean'] = train_mix['_'.join(groupcols)+'_mean'].astype(np.float16)

    train_mix.drop(cols_new, axis=1, inplace = True)
    train_mix.fillna({'_'.join(groupcols)+'_mean':0}, inplace = True)
    
    return train_mix


def remove_duplicate(sales_train, test):
    # Якутск Орджоникидзе, 56
    sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
    test.loc[test.shop_id == 0, 'shop_id'] = 57
    # Якутск ТЦ "Центральный"
    sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
    test.loc[test.shop_id == 1, 'shop_id'] = 58
    # Жуковский ул. Чкалова 39м²
    sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
    test.loc[test.shop_id == 10, 'shop_id'] = 11
    #
    sales_train.loc[sales_train.shop_id == 39, 'shop_id'] = 40
    test.loc[test.shop_id == 39, 'shop_id'] = 40
	
	return sales_train, test


def preprocess_sales_train(df):
    
    df = df[df['item_cnt_day'] < 1000]
    df = df[df['item_price'] < 100000]
    df = df[df.item_price > 0].reset_index(drop=True)
    df.loc[df.item_cnt_day < 1, 'item_cnt_day'] = 0
    
    return df


def name_correction(x):
    x = x.lower()
    x = re.sub("[']+", '', x)
    x = re.sub("[-]+", ' ', x)
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x


def select_value(row, lags, new_col):
    for i in lags:
        if np.isnan(row[f'{new_col}_lag_'+str(i)]):
            continue
        if row[f'{new_col}_lag_'+str(i)]:
            return row[f'{new_col}_lag_'+str(i)]
    return 0


def select_trend(row):
    for i in lags:
        if np.isnan(row['delta_price_lag_'+str(i)]):
            continue
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0


def prepare_data(path):
    
    sales_train = pd.read_csv(f'{path}/sales_train_v2.csv')
    test = pd.read_csv(f'{path}/test.csv')
    item_cat_path = f'{path}/item_categories.csv'
    item_path = f'{path}/items.csv'
    shops_path = f'{path}/shops.csv'
    
    sales_train, test = remove_duplicate(sales_train, test)
    
    sales_train = preprocess_sales_train(sales_train)
    median = sales_train[(sales_train.shop_id==32)&(sales_train.item_id==2973)&(sales_train.date_block_num==4)&(sales_train.item_price>0)].item_price.median()
    sales_train.loc[sales_train.item_price<0, 'item_price'] = median
    
    shops = process_shop(shops_path)
    item = process_item(item_path)
    item_cat = process_item_cat(item_cat_path)
    
    sales_train['revenue'] = sales_train['item_price'] *  sales_train['item_cnt_day']
    
    cols = ['date_block_num','shop_id','item_id']
    
    test['date_block_num']= 34
    test['date_block_num'] = test['date_block_num'].astype(np.int8)
    test['shop_id'] = test['shop_id'].astype(np.int8)
    test['item_id'] = test['item_id'].astype(np.int16)
    test = test[cols]
    
    sales_group = sales_train.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
                                        'item_cnt_day': ['sum'],
                                        'revenue': ['sum']
    })
    
    sales_group.columns = ['_'.join(col).strip() for col in sales_group.columns.values]
    sales_group.rename(columns = {
                                'revenue_sum': 'revenue_month',
                                'item_cnt_day_sum': 'item_cnt_month',
                                }, inplace = True) 
    sales_group.reset_index(inplace=True)
    
    cols = [col for col in item.columns.values if col not in ['item_name', 'name_copy']]
    sales_group = pd.merge(sales_group, item[cols], on = ['item_id'], how = 'left')
    
    test['category'] = 4
    
    shop_train = sales_group.loc[sales_group['date_block_num']<=33]['shop_id'].unique()
    item_lst = sales_group.loc[sales_group['date_block_num']<=33]['item_id'].unique()
    shop_test = test.loc[test['category']==0]['shop_id'].unique()
    
    for shop in shop_train:
        item_lst_shop = sales_group.loc[(sales_group['date_block_num']<=33)&(sales_group['shop_id'] == shop)]['item_id'].unique()
        test.loc[(test['shop_id'] == shop) & (test['item_id'].isin(item_lst_shop)), 'category'] = 0
        
    test.loc[~(test['item_id'].isin(item_lst)), 'category'] = 2
    test.loc[test['category'] == 4, 'category'] = 1
    
    cols = ['date_block_num','shop_id','item_id']
    train_mix = []
    for i in range(34):
        sales = sales_train[sales_train.date_block_num==i]
        train_mix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    train_mix = pd.DataFrame(np.vstack(train_mix), columns=cols)
    train_mix['date_block_num'] = train_mix['date_block_num'].astype(np.int8)
    train_mix['shop_id'] = train_mix['shop_id'].astype(np.int8)
    train_mix['item_id'] = train_mix['item_id'].astype(np.int16)
    train_mix.sort_values(cols,inplace=True)
    gc.collect()
    
    train_mix['category'] = 4
    train_mix = pd.concat([train_mix, test], ignore_index=True, sort=False, keys=cols)
    train_mix.fillna(0, inplace=True)
    
    train_mix = target_generate(train_mix)
    train_mix = merge_dataset(train_mix)
    
    train_mix = lag_feature(train_mix,[1,2,3,6],['shop_id','item_id'],'revenue_month')
        
    for shop in shop_train:
        item_lst_shop = sales_group.loc[(sales_group['date_block_num']<=33)&(sales_group['shop_id'] == shop)]['item_id'].unique()
        train_mix.loc[(train_mix['shop_id'] == shop) & (train_mix['item_id'].isin(item_lst_shop)), 'category'] = 0
        
    train_mix.loc[~(train_mix['item_id'].isin(item_lst)), 'category'] = 2
    train_mix.loc[train_mix['category'] == 4, 'category'] = 1
    
    train_mix = lag_feature(train_mix,[1,2,3,6,12],[],'item_cnt_month')    
    train_mix = lag_feature(train_mix,[1,2,3,6],[],'revenue_month')
    train_mix = lag_feature(train_mix,[1,2,3,6],['item_id'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,6],['shop_id'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['name_1'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['name_2'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['name_3'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['name_4'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['shop_id','name_1'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['shop_id','name_2'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['shop_id','name_3'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,12],['shop_id','name_4'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3],['shop_id'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['item_category_id'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['item_category_id'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','item_category_id'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','item_category_id'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','type_code'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','type_code'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','subtype_code'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','subtype_code'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['shop_city'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,6,12],['shop_city','item_id'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1,2,3,6,12],['type_code'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['type_code'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['subtype_code'],'item_cnt_month')
    train_mix = lag_feature(train_mix,[1],['subtype_code'],'revenue_month')
    train_mix = lag_feature(train_mix,[1],['shop_id','item_id'],'delta_revenue','no')
    train_mix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
    
    gc.collect()
    
    group = sales_train.groupby(['item_id']).agg({'item_price': ['mean']})
    group.columns = ['item_avg_item_price']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on=['item_id'], how='left')
    train_mix['item_avg_item_price'] = train_mix['item_avg_item_price'].astype(np.float16)

    group = sales_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_item_price']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on=['date_block_num','item_id'], how='left')
    train_mix['date_item_avg_item_price'] = train_mix['date_item_avg_item_price'].astype(np.float16)

    train_mix = lag_feature(train_mix,[1,2,3,4,5,6,7,8,9,10,11,12],['item_id'],'date_item_avg_item_price')
    new_col = '_'.join(['item_id'])+'_avg_'+str('date_item_avg_item_price')

    for i in lags:
        train_mix['delta_price_lag_'+str(i)] =             (train_mix[f'{new_col}_lag_'+str(i)] - train_mix['item_avg_item_price']) / train_mix['item_avg_item_price']

    train_mix['delta_price_lag'] = train_mix.apply(select_trend, axis=1)
    train_mix['delta_price_lag'] = train_mix['delta_price_lag'].astype(np.float16)
    train_mix['delta_price_lag'].fillna(0, inplace=True)

    features_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
    for i in lags:
        features_to_drop += [f'{new_col}_lag_'+str(i)]
        features_to_drop += ['delta_price_lag_'+str(i)]
    train_mix.drop(features_to_drop, axis=1, inplace=True)
    
    group = sales_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_item_avg_price']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on=['date_block_num','item_id'], how='left')
    train_mix['date_item_avg_price'] = train_mix['date_item_avg_price'].astype(np.float16)

    train_mix = lag_feature(train_mix,[1,2,3,4,5,6,7,8,9,10,11,12],['item_id'],'date_item_avg_price','no')
    new_col = '_'.join(['item_id'])+'_avg_'+str('date_item_avg_price')
    train_mix['last_item_avg_price'] = train_mix.apply(select_value, axis=1)
    train_mix['last_item_avg_price'] = train_mix['last_item_avg_price'].astype(np.float16)
    train_mix['last_item_avg_price'].fillna(0, inplace=True)

    gc.collect()

    features_to_drop = ['date_item_avg_price']
    for i in lags:
        features_to_drop += [f'{new_col}_lag_'+str(i)]
    train_mix.drop(features_to_drop, axis=1, inplace=True)
    
    group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_price': ['mean']})
    group.columns = ['date_shop_item_avg_price']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on=['date_block_num','shop_id','item_id'], how='left')
    train_mix['date_shop_item_avg_price'] = train_mix['date_shop_item_avg_price'].astype(np.float16)

    train_mix = lag_feature(train_mix,[1,2,3,4,5,6,7,8,9,10,11,12],['shop_id','item_id'],'date_shop_item_avg_price','no')
    new_col = '_'.join(['shop_id','item_id'])+'_avg_'+str('date_shop_item_avg_price')
    train_mix['last_shop_item_avg_price'] = train_mix.apply(select_value, axis=1)
    train_mix['last_shop_item_avg_price'] = train_mix['last_shop_item_avg_price'].astype(np.float16)
    train_mix['last_shop_item_avg_price'].fillna(0, inplace=True)

    gc.collect()

    fetures_to_drop = ['date_shop_item_avg_price']
    for i in lags:
        fetures_to_drop += [f'{new_col}_lag_'+str(i)]
    train_mix.drop(fetures_to_drop, axis=1, inplace=True)
    
    train_mix['last_price_diff'] = train_mix['last_shop_item_avg_price'] - train_mix['last_item_avg_price']
    train_mix['last_price_ratio'] = train_mix['last_shop_item_avg_price'] / train_mix['last_item_avg_price']
    train_mix['last_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)
    
    group = sales_train.groupby(['item_id']).agg({'item_price': ['max']})
    group.columns = ['item_max_price']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on=['item_id'], how='left')
    train_mix['item_max_price'] = train_mix['item_max_price'].astype(np.float16)

    train_mix.fillna(0, inplace = True)
    del group
    
    train_mix['last_shop_item_city_item_price_ratio'] = train_mix['last_shop_item_avg_price'] / train_mix['last_city_item_avg_price']
    train_mix['last_shop_item_city_item_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_shop_item_cat_price_ratio'] = train_mix['last_shop_item_avg_price'] / train_mix['last_cat_avg_price']
    train_mix['last_shop_item_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_shop_item_city_cat_price_ratio'] = train_mix['last_shop_item_avg_price'] / train_mix['last_city_cat_avg_price']
    train_mix['last_shop_item_city_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_shop_item_shop_cat_price_ratio'] = train_mix['last_shop_item_avg_price'] / train_mix['last_shop_cat_avg_price']
    train_mix['last_shop_item_shop_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_item_cat_price_ratio'] = train_mix['last_item_avg_price'] / train_mix['last_cat_avg_price']
    train_mix['last_item_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_item_shop_cat_price_ratio'] = train_mix['last_item_avg_price'] / train_mix['last_shop_cat_avg_price']
    train_mix['last_item_shop_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_item_city_cat_price_ratio'] = train_mix['last_item_avg_price'] / train_mix['last_city_cat_avg_price']
    train_mix['last_item_city_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_item_city_item_price_ratio'] = train_mix['last_item_avg_price'] / train_mix['last_city_item_avg_price']
    train_mix['last_item_city_item_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)

    train_mix['last_shop_cat_city_cat_price_ratio'] = train_mix['last_shop_cat_avg_price'] / train_mix['last_city_cat_avg_price']
    train_mix['last_shop_cat_city_cat_price_ratio'].replace([np.inf, -np.inf], 0, inplace = True)
    
    cols = ['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_lag_6', 'item_cnt_month_lag_12']
    train_mix['shop_item_avg_cnt'] = train_mix[cols].mean(axis=1)
    train_mix['shop_item_max_cnt'] = train_mix[cols].max(axis=1)
    
    cols = ['date_item_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3',
            'date_item_avg_item_cnt_lag_6','date_item_avg_item_cnt_lag_12']
    train_mix['item_avg_cnt'] = train_mix[cols].mean(axis=1)
    train_mix['item_max_cnt'] = train_mix[cols].max(axis=1)
    
    train_mix['pred_last_3_shop_item_cnt_month_dif'] = train_mix['item_cnt_month_lag_1']*2 - train_mix['item_cnt_month_lag_2']
    train_mix['pred_last_3_shop_item_cnt_month_ratio'] = train_mix['item_cnt_month_lag_1']*train_mix['item_cnt_month_lag_1'] / train_mix['item_cnt_month_lag_2']
    train_mix.fillna(0, inplace = True)
    train_mix.replace([np.inf,-np.inf],0,inplace=True)
    train_mix.loc[train_mix['pred_last_3_shop_item_cnt_month_dif']<0,'pred_last_3_shop_item_cnt_month_dif']=0
    
    train_mix['pred_last_3_item_cnt_month_dif'] = train_mix['date_item_avg_item_cnt_lag_1']*2 - train_mix['date_item_avg_item_cnt_lag_2']
    train_mix['pred_last_3_item_cnt_month_ratio'] = train_mix['date_item_avg_item_cnt_lag_1']*train_mix['date_item_avg_item_cnt_lag_1'] / train_mix['date_item_avg_item_cnt_lag_2']
    train_mix.fillna(0, inplace = True)
    train_mix.replace([np.inf,-np.inf],0,inplace=True)
    train_mix.loc[train_mix['pred_last_3_item_cnt_month_dif']<0,'pred_last_3_item_cnt_month_dif']=0
    
    train_mix['pred_last_3_shop_item_cnt_month_ratio_2'] = train_mix['item_cnt_month_lag_1']*train_mix['date_item_avg_item_cnt_lag_1'] / train_mix['date_item_avg_item_cnt_lag_2']
    cols = ['item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3']
    train_mix['pred_last_3_shop_item_cnt_month_ratio_3'] = train_mix[cols].mean(axis=1)*train_mix['date_item_avg_item_cnt_lag_1'] / train_mix['date_item_avg_item_cnt_lag_2']
    train_mix.fillna(0, inplace = True)
    
    train_mix.fillna(0, inplace = True)
    
    group = sales_train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)

    train_mix = pd.merge(train_mix, group, on=['date_block_num','shop_id'], how='left')
    train_mix['date_shop_revenue'] = train_mix['date_shop_revenue'].astype(np.float32)

    group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
    group.columns = ['shop_avg_revenue']
    group.reset_index(inplace=True)

    train_mix = pd.merge(train_mix, group, on=['shop_id'], how='left')
    train_mix['shop_avg_revenue'] = train_mix['shop_avg_revenue'].astype(np.float32)

    train_mix['delta_revenue'] = (train_mix['date_shop_revenue'] - train_mix['shop_avg_revenue']) / train_mix['shop_avg_revenue']
    train_mix['delta_revenue'] = train_mix['delta_revenue'].astype(np.float16)
    
    train_mix['month'] = train_mix['date_block_num'] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    train_mix['days'] = train_mix['month'].map(days).astype(np.int8)
    
    cache = {}
    train_mix['item_shop_last_sale'] = -1
    train_mix['item_shop_last_sale'] = train_mix['item_shop_last_sale'].astype(np.int8)
    for idx, row in train_mix.iterrows():    
        key = str(row.item_id)+' '+str(row.shop_id)
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            train_mix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num  
            
    cache = {}
    train_mix['item_last_sale'] = -1
    train_mix['item_last_sale'] = train_mix['item_last_sale'].astype(np.int8)
    for idx, row in train_mix.iterrows():    
        key = row.item_id
        if key not in cache:
            if row.item_cnt_month!=0:
                cache[key] = row.date_block_num
        else:
            last_date_block_num = cache[key]
            if row.date_block_num>last_date_block_num:
                train_mix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
                cache[key] = row.date_block_num     
                
    train_mix['category'] = train_mix['category'].astype(np.int8)
    train_mix['name_1'] = train_mix['name_1'].astype(np.int16)
    train_mix['name_2'] = train_mix['name_2'].astype(np.int16)
    train_mix['name_3'] = train_mix['name_3'].astype(np.int16)
    train_mix['name_4'] = train_mix['name_4'].astype(np.int16)
    train_mix['name_5'] = train_mix['name_5'].astype(np.int16)
    
    train_mix['language'] = train_mix['language'].astype(np.int8)
    train_mix['isbd'] = train_mix['isbd'].astype(np.int8)
    train_mix['isрегион'] = train_mix['isрегион'].astype(np.int8)
    train_mix['isjewel'] = train_mix['isjewel'].astype(np.int8)
    train_mix['isdigital'] = train_mix['isdigital'].astype(np.int8)
    train_mix['console'] = train_mix['console'].astype(np.int8)
    train_mix['has_cd'] = train_mix['has_cd'].astype(np.int8)
    train_mix['has_dvd'] = train_mix['has_dvd'].astype(np.int8)
    train_mix['has_mp3'] = train_mix['has_mp3'].astype(np.int8)
    train_mix['has_lp'] = train_mix['has_lp'].astype(np.int8)
    train_mix['version'] = train_mix['version'].astype(np.int8)
    
    group = sales_group.groupby(['item_id','shop_id']).agg({'date_block_num': 'min'})
    group.columns = ['item_shop_month1']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on = ['item_id', 'shop_id'], how ='left')

    train_mix['item_shop_first_sale'] = train_mix['date_block_num'] - train_mix['item_shop_month1']
    train_mix.drop('item_shop_month1', axis = 1, inplace = True)

    group = sales_group.groupby('item_id').agg({'date_block_num': 'min'})
    group.columns = ['item_month1']
    group.reset_index(inplace = True)
    train_mix = pd.merge(train_mix, group, on = ['item_id'], how = 'left')

    train_mix['item_first_sale'] = train_mix['date_block_num'] - train_mix['item_month1']
    train_mix.drop('item_month1', axis = 1, inplace = True)

    del group
    gc.collect()
    
    train_mix['item_shop_first_sale'] = train_mix['date_block_num'] - train_mix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
    train_mix['item_first_sale'] = train_mix['date_block_num'] - train_mix.groupby('item_id')['date_block_num'].transform('min')
    
    group = sales_group.groupby(['name_1','shop_id']).agg({'date_block_num': 'min'})
    group.columns = ['n1_shop_month1']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on = ['name_1', 'shop_id'], how ='left')

    train_mix['n1_shop_first_sale'] = train_mix['date_block_num'] - train_mix['n1_shop_month1']
    train_mix.drop('n1_shop_month1', axis = 1, inplace = True)

    group = sales_group.groupby(['name_2','shop_id']).agg({'date_block_num': 'min'})
    group.columns = ['n2_shop_month1']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on = ['name_2', 'shop_id'], how ='left')

    train_mix['n2_shop_first_sale'] = train_mix['date_block_num'] - train_mix['n2_shop_month1']
    train_mix.drop('n2_shop_month1', axis = 1, inplace = True)

    group = sales_group.groupby(['name_1']).agg({'date_block_num': 'min'})
    group.columns = ['n1_month1']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on = ['name_1'], how ='left')

    train_mix['n1_first_sale'] = train_mix['date_block_num'] - train_mix['n1_month1']
    train_mix.drop('n1_month1', axis = 1, inplace = True)

    group = sales_group.groupby(['name_2']).agg({'date_block_num': 'min'})
    group.columns = ['n2_month1']
    group.reset_index(inplace=True)
    train_mix = pd.merge(train_mix, group, on = ['name_2'], how ='left')

    train_mix['n2_first_sale'] = train_mix['date_block_num'] - train_mix['n2_month1']
    train_mix.drop('n2_month1', axis = 1, inplace = True)
    
    train_mix['first_sale_diff'] = train_mix['item_first_sale'] - train_mix['item_shop_first_sale']
    train_mix['isChristmas'] = 0
    train_mix.loc[(train_mix['date_block_num']%12==11),'isChristmas']=1
    train_mix['interval_last_Christmas'] = 11 - (train_mix['date_block_num']%12)
    train_mix['interval_first_Christmas'] = (train_mix['date_block_num']%12) + 1
    train_mix['item_new'] = 0
    train_mix.loc[(train_mix['item_first_sale']>=0)&(train_mix['item_first_sale']<=1),'item_new'] = 1
    train_mix['item_shop_new'] = 0
    train_mix.loc[(train_mix['item_shop_first_sale']>=0)&(train_mix['item_shop_first_sale']<=1),'item_shop_new'] = 1
    
    train_mix.fillna({'n1_first_sale': 0,
                      'n2_first_sale': 0},
                    inplace = True)
    
    train_mix.fillna({'n1_shop_first_sale': -100,
                      'n2_shop_first_sale': -100},
                    inplace = True)
    
    train_mix['n1_new'] = 0
    train_mix.loc[(train_mix['n1_first_sale']>=0)&(train_mix['n1_first_sale']<=1),'n1_new'] = 1
    train_mix['n2_new'] = 0
    train_mix.loc[(train_mix['n2_first_sale']>=0)&(train_mix['n2_first_sale']<=1),'n2_new'] = 1

    train_mix['n1_shop_new'] = 0
    train_mix.loc[(train_mix['n1_shop_first_sale']>=0)&(train_mix['n1_shop_first_sale']<=1),'n1_shop_new'] = 1
    train_mix['n2_shop_new'] = 0
    train_mix.loc[(train_mix['n2_shop_first_sale']>=0)&(train_mix['n2_shop_first_sale']<=1),'n2_shop_new'] = 1
    
    train_mix['shop_since_open'] = train_mix['date_block_num'] - train_mix.groupby(['shop_id'])['date_block_num'].transform('min')
    
    group = train_mix.loc[train_mix['date_block_num']<=33].groupby(['shop_id']).agg({'date_block_num':'max'})
    group.columns = ['max_date']
    group.reset_index(inplace=True)

    train_mix = pd.merge(train_mix, group, on=['shop_id'], how='left')
    train_mix['max_date'] = train_mix['max_date'].astype(np.int8)
    
    train_mix['shop_duration'] = train_mix['max_date'] - train_mix['shop_since_open']
    
    train_mix['isChristmas'] = train_mix['isChristmas'].astype(np.int8)
    train_mix['item_new'] = train_mix['item_new'].astype(np.int8)
    train_mix['item_shop_new'] = train_mix['item_shop_new'].astype(np.int8)
    train_mix['max_date'] = train_mix['max_date'].astype(np.int8)
    train_mix['shop_duration'] = train_mix['shop_duration'].astype(np.int8)
    train_mix['n1_new'] = train_mix['n1_new'].astype(np.int8)
    train_mix['n2_new'] = train_mix['n2_new'].astype(np.int8)
    train_mix['n1_shop_new'] = train_mix['n1_shop_new'].astype(np.int8)
    train_mix['n2_shop_new'] = train_mix['n2_shop_new'].astype(np.int8)
    train_mix['n1_shop_first_sale'] = train_mix['n1_shop_first_sale'].astype(np.int8)
    train_mix['n2_shop_first_sale'] = train_mix['n2_shop_first_sale'].astype(np.int8)
    train_mix['n1_first_sale'] = train_mix['n1_first_sale'].astype(np.int8)
    train_mix['n2_first_sale'] = train_mix['n2_first_sale'].astype(np.int8)
    
    train_mix.loc[train_mix['delta_price_lag']==-1, 'delta_price_lag'] = 0
    
    cols = [col for col in item.columns.values if col not in ['name_2','name_3','name_4','name_5','item_category_id']]
    train_mix = pd.merge(train_mix, item[cols], on=['item_id'], how='left')
    
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_3','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','version','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','version','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','console','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','console','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','language','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','language','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','isрегион','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','isрегион','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','isjewel','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','isjewel','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','isdigital','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','isdigital','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','isbd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','isbd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','has_cd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','has_cd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','has_dvd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','has_dvd','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','has_mp3','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','has_mp3','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_1','subtype_code','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_id','name_2','subtype_code','item_new'], 'item_cnt_month')
    train_mix = FE_lag_statistic(train_mix, [1,2,3,4,5,6,7,8,9,10,11,12,13,14], ['shop_city','item_category_id','console','item_new'], 'item_cnt_month')
    
    train_mix = reduce_mem_usage(train_mix)
    
    return train_mix


if __name__ == "__main__":
    
    path = '../input'
    
    train_mix = prepare_data(path)
    
    with open('../processed/train_mix.pickle', 'wb') as handle:
        pickle.dump(train_mix, handle, protocol = pickle.HIGHEST_PROTOCOL)