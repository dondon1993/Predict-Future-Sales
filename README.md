# Predict-Future-Sales: Overview
 
Kaggle Playground competiton: https://www.kaggle.com/c/competitive-data-science-predict-future-sales

This challenge serves as final project for the "How to win a data science competition" Coursera course.

In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company. Competition provides sale history data from Russian shops over past two years (0 - 33rd month) and asks to predict the number of each item sold in the 34th month. Besides the sale history, datasets containing the information of items, item categories and shops are also provided. The prediction is evaluated by metric mean-squared-error (MSE). True target values are clipped into [0,20] range.

# Data Description

You are provided with daily historical sales data. The task is to forecast the total amount of products sold in every shop for the test set. Note that the list of shops and products slightly changes every month. Creating a robust model that can handle such situations is part of the challenge.

## File descriptions

* sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
* test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
* sample_submission.csv - a sample submission file in the correct format.
* items.csv - supplemental information about the items/products.
* item_categories.csv - supplemental information about the items categories.
* shops.csv- supplemental information about the shops.

## Data fields

* ID - an Id that represents a (Shop, Item) tuple within the test set
* shop_id - unique identifier of a shop
* item_id - unique identifier of a product
* item_category_id - unique identifier of item category
* item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
* item_price - current price of an item
* date - date in format dd/mm/yyyy
* date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
* item_name - name of item
* shop_name - name of shop
* item_category_name - name of item category

# Exploratory Data Analysis

Items shown in the training set and test set consist of video games, game consoles, movies, music CDs, books and souveniors, all of which can be found in a bookstore. Item names are important since video games or movies usually have a series of a topic, prequel, sequel (Assasin's creed 1, 2, 3) and different versions (gold, premium, legendary).

Similary, item_category or shop names also contain information and can be decomposed to extract features.

Items in the test set can be divided into three different types. 
* Category 1: Item and shop has sale history.
* Category 2: Item has sale history but not for a specific shop.
* Category 3: New items only showing up in the test set.
* They key is to find the best way to Model items in each category.

# Data Preparation

Provided sale history contains daily sale record. Since it only records when a sale happens, therefore all records are non-zero. Therefore, to predict the count of an item sold in a month, we need to aggregate the sale history dataset to form monthly sale record padded with 0 sale record. Follow public kernals, the training set can be created using product of all shops and items sold in a month.

Then the training set will be merged with item, item category and shop dataset so that we can use relevant features contained in these data sets.

# Feature Engineering

* Label encoding categorical features.
* Optimization of data types to save space.
* Various lag + aggregation features. For example, df.groupby([month,item]).agg({sale_count: ["sum","mean"]}).
* The shop open and closed month.
* Item first sale and last sale month.
* Price change features.
* Time features such as month and whether this month is Christmas(Christmas boost sale most significantly).

# Modeling

When modeling category 0 and 1 data, data from month 33 is used as the validation set.

Most of time category 0 and 1 are trained together using the generated training set. lgbm, xgboost and neural network are trained. Neural network are also trained only on category 0 data. For neural network, it's found that using embeddings to encode categorical features can improve neural network model. But the best performance of neural network is still the same as lgbm or xgboost. Tuning hyperparameters of models and blending lgbm, xgboost and neural network result in 0.01 increase of the metric.

The largest improvement comes from modeling category 2 data which consists items without sale history. With models trained on category 0 and 1 data, it is found that the encoded number of item name is beneficial to category 2 data prediction and the prediction are all small. The small prediction for category 2 is due to model of category 0 and 1 depends heavily on lag features, which category 2 data doesn't have. The encoded number of item name is beneficial to the model because items with similiar encoded numbers are similar to each other (for example, Assasin's Creed 1 and Assasin's Creed 2). Therefore the encoded numbers accidentally capture the similiarity between items.

Therefore, the question is what is the right training dataset for category 2? And is there any history to use? The answer is yes. First, only data representing items newly showing up in the market are sampled to form the training set. There is a feature called item_first_sale, which represents the time since an item shows up in the market, and we only sample the data with item_first_sale = 0 or 1.

Now to use the lag feature of new items, we have to find out how a new item is related to the items that are already in the market. This relation lies in the name of the item. Take Assasin's Creed as an example again, if Assasin's Creed 1 is popular, we can expect Assasin's Creed 2 will be popular as well. And to use this correlation, we can simply use the first one or two words in the item name as a feature and create more lag features based on them.

Lgbm is trained on the new item dataset with 5 fold w/o shuffle cross-validation scheme. After hyperparameters tuning and some feature selection, a 0.04 increase of the metric is achieved. The result final lands on rank 12 out of 6055 by the time this document is written.

# Failures

* In this dataset, the most tricky part comes from data leakage and overfitting. A lot of engineered features can lead to them. For example, df.groupby(['shop']).({sale_count: ["sum","mean"]}). 
* Target encoding doesn't work in my case.
* Feature selection using PCA, pearson, k2_samp.
* Data resampling to align the training set and test set with all categories based on feature distributions.

# Things to improve

* More proficient EDA techniques
* Use more sophisticated methods to find the correlation between different items. Since we are dealing with a lot of strings showing up in the item names, item category names or shop names, maybe we can use sequence model to better capture the meaning of names.
* Data sampling. How to better construct the training dataset and validation scheme.
* Various feature engineering techniques.
