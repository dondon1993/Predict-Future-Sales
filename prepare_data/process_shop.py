import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def process_shop(path):
    
    shops = pd.read_csv(path)
    
    shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
    shops['city'] = shops['shop_name'].str.split(' ').map(lambda x:x[0])
    shops['category'] = shops['shop_name'].str.split(' ').map(lambda x:x[1]).astype(str)

    shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

    category = ['Орджоникидзе,', 'ТЦ', 'ТРК', 'ТРЦ','ул.', 'Магазин', 'ТК', 'склад']
    shops.category = shops.category.apply(lambda x: x if (x in category) else 'etc')
    
    category = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК']
    shops.category = shops.category.apply(lambda x: x if (x in category) else 'etc')
    
    shops['shop_city'] = shops.city
    shops['shop_category'] = shops.category

    shops['shop_city'] = LabelEncoder().fit_transform(shops['shop_city'])
    shops['shop_category'] = LabelEncoder().fit_transform(shops['shop_category'])

    shops = shops[['shop_id','shop_city', 'shop_category']]
    
    return shops