import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_item_cat(path):
    
    item_cat = pd.read_csv(path)
    
    item_cat['type_code'] = item_cat.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
    item_cat.loc[(item_cat.type_code == 'Игровые') | (item_cat.type_code == 'Аксессуары'), 'category'] = 'Игры'
    item_cat.loc[item_cat.type_code == 'PC', 'category'] = 'Музыка'

    category = ['Игры', 'Карты', 'Кино', 'Книги','Музыка', 'Подарки', 'Программы', 'Служебные', 'Чистые']

    item_cat['type_code'] = item_cat.type_code.apply(lambda x: x if (x in category) else 'etc')

    item_cat['split'] = item_cat.item_category_name.apply(lambda x: x.split('-'))
    item_cat['subtype'] = item_cat['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
    
    item_cat['type_code'] = LabelEncoder().fit_transform(item_cat['type_code'])
    item_cat['subtype_code'] = LabelEncoder().fit_transform(item_cat['subtype'])
    item_cat = item_cat[['item_category_id','type_code', 'subtype_code']]
    
    l_cat = list(item_cat.item_category_id)
    
    for ind in range(0,1):
         l_cat[ind] = 'PC Headsets / Headphones'

    for ind in range(1,8):
         l_cat[ind] = 'Access'

    l_cat[8] = 'Tickets (figure)'

    l_cat[9] = 'Delivery of goods'

    for ind in range(10,18):
        l_cat[ind] = 'Consoles'

    for ind in range(18,25):
        l_cat[ind] = 'Consoles Games'

    l_cat[25] = 'Accessories for games'

    for ind in range(26,28):
        l_cat[ind] = 'phone games'

    for ind in range(28,32):
        l_cat[ind] = 'CD games'

    for ind in range(32,37):
        l_cat[ind] = 'Card'

    for ind in range(37,43):
        l_cat[ind] = 'Movie'

    for ind in range(43,55):
        l_cat[ind] = 'Books'

    for ind in range(55,61):
        l_cat[ind] = 'Music'

    for ind in range(61,73):
        l_cat[ind] = 'Gifts'

    for ind in range(73,79):
        l_cat[ind] = 'Soft'

    for ind in range(79,81):
        l_cat[ind] = 'Office'

    for ind in range(81,83):
        l_cat[ind] = 'Clean'

    l_cat[83] = 'Elements of a food'
    
    lb = LabelEncoder()
    item_cat['item_cat_id_fix'] = lb.fit_transform(l_cat)
    
    return item_cat