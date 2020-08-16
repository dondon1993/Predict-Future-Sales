import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_item(path):
    
    item = pd.read_csv(path)
    
    item['name_copy'] = item['item_name'].apply(lambda x: name_correction(x))
    
    item['name_copy'] = item.name_copy.str.replace('xbox360', 'xbox 360')
    item['name_copy'] = item.name_copy.str.replace('x360', 'xbox 360')
    item['name_copy'] = item.name_copy.str.replace('xone', 'xbox one')
    item['name_copy'] = item.name_copy.str.replace('xboxone', 'xbox one')
    
    word_list = item['name_copy'].str.split(' ')
    item['name_1'] = word_list.str[0]
    item['name_2'] = word_list.str[0] + " " + word_list.str[1]
    item['name_3'] = word_list.str[0] + " " + word_list.str[1] + " " +word_list.str[2]
    item['name_4'] = word_list.str[0] + " " + word_list.str[1] + " " +word_list.str[2] + " " +word_list.str[3]
    item['name_5'] = word_list.str[0] + " " + word_list.str[1] + " " +word_list.str[2] + " " +word_list.str[3] + " " +word_list.str[4]
    
    item.loc[item['name_2'].isnull(),'name_2'] = item.loc[item['name_2'].isnull(),'name_1']
    item.loc[item['name_3'].isnull(),'name_3'] = item.loc[item['name_3'].isnull(),'name_2']
    item.loc[item['name_4'].isnull(),'name_4'] = item.loc[item['name_4'].isnull(),'name_3']
    item.loc[item['name_5'].isnull(),'name_5'] = item.loc[item['name_5'].isnull(),'name_4']
    
    item['name_1'] = item['name_1'].astype('str')
    item['name_2'] = item['name_2'].astype('str')
    item['name_3'] = item['name_3'].astype('str')
    item['name_4'] = item['name_4'].astype('str')
    item['name_5'] = item['name_5'].astype('str')
    
    item['language'] = -1
    item.loc[item['name_copy'].str.contains('русс'),'language'] = 0
    item.loc[item['name_copy'].str.contains('англий'), 'language']=1
    item['language'] = item['language'].astype(np.int8)
    
    item['isbd'] = 0
    item.loc[item['name_copy'].str.contains('bd'),'isbd'] = 1
    item['isрегион'] = 0
    item.loc[item['name_copy'].str.contains('регион'),'isрегион'] = 1
    item['isjewel'] = 0
    item.loc[item['name_copy'].str.contains('jewel'),'isjewel'] = 1
    item['isdigital'] = 0
    item.loc[(item['name_copy'].str.contains('digi'))|(item['name_copy'].str.contains('цифро')),'isdigital']=1
    
    item['console'] = -1
    consoles = ['pc', 'xbox', 'xbox 360', 'xbox one', 'ps', 'ps3', 'ps4']
    count = 0
    for console in consoles:
        item.loc[item['name_copy'].str.contains(console),'console'] = count
        count = count + 1
        
    formats = ['cd', 'dvd', 'mp3', 'lp']
    for format in formats:
        feature = f'has_{format}'
        item[feature] = 0
        item.loc[item['name_copy'].str.contains(format),feature]=1
        
    versions = []
    versions.append(['basic','essential','standard'])
    versions.append(['classics','exclusive','special','season','специальное'])
    versions.append(['deluxe', 'limited'])
    versions.append(['gold edition','premium'])
    versions.append(['ultimate edition'])
    versions.append(['collector', 'legendary', 'коллекционное'])
    versions.append(['допол','add','dlc', 'pack', 'расширенное'])
    versions.append(['пред']) # pre-order

    for i in range(8):
        item.loc[(item['item_category_id']>=18) 
                 & (item['item_category_id']<32) 
                 & (item['name_copy'].str.contains('|'.join(versions[i])))
                 & (item['version']==-1), 'version'] = i

    item.loc[item['version']==-1, 'version'] = 0
    
    item['name_1'] = LabelEncoder().fit_transform(item['name_1'])
    item['name_2'] = LabelEncoder().fit_transform(item['name_2'])
    item['name_3'] = LabelEncoder().fit_transform(item['name_3'])
    item['name_4'] = LabelEncoder().fit_transform(item['name_4'])
    item['name_5'] = LabelEncoder().fit_transform(item['name_5'])
    item.drop(['item_name', 'name_copy'], axis=1, inplace=True)
    
    return item