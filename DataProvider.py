import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data():
    '''Preprocess data and return X, y, label_list'''
    
    # Data files
    acc_x_card = 'data/tj_02_acc_x_card.csv'
    acc_transc = 'data/tj_02_account_transaction.csv'
    cred_transc = 'data/tj_02_creditcard_transaction.csv'
    train_file = 'data/tj_02_training.csv'
    test_file = 'data/tj_02_test.csv'
    
    df_acc_x = pd.read_csv(acc_x_card)
    df_acc_trancs = pd.read_csv(acc_transc)
    df_cred = pd.read_csv(cred_transc)
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)
    df_train.columns = ['account_no', 'label']
    df_test.columns = ['account_no']

    # Preprocess account transaction data
    dummy_type = pd.get_dummies(df_acc_trancs['txn_type'])
    df = pd.concat([df_acc_trancs, dummy_type], axis=1)
    df = df.drop(['txn_type'], axis=1)

    df['hour'] = df['txn_hour']//3
    df_hour = pd.get_dummies(df['hour'])
    df_hour.columns = ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
    df = pd.concat([df, df_hour], axis=1)
    df = df.drop(['txn_hour', 'hour'], axis=1)

    df['date'] = pd.to_datetime(df['txn_dt'])
    df['day'] = df['date'].dt.dayofweek
    df['weekend'] = df['day'] >= 5
    df['weekday'] = df['day'] < 5
    df['weekday'] = df['weekday'].astype(int)

    holidays = pd.read_csv('../../Holiday2016.csv', header=None)
    df['holidays'] = df['txn_dt'].isin(holidays[0].tolist())
    df['holidays'] = df['holidays'] | df['weekend']
    df['holidays'] = df['holidays'].astype(int)
    df['weekend'] = df['weekend'].astype(int)
    df = df.drop(['txn_dt', 'day'], axis=1)
    
    df = df[df['txn_amount'] < df_acc_trancs.quantile(0.995)['txn_amount']]

    txn_am_mean = df.groupby(['account_no'])['txn_amount'].mean()
    txn_am_std = df.groupby(['account_no'])['txn_amount'].std()
    txn_am_max = df.groupby(['account_no'])['txn_amount'].max()
    txn_am_min = df.groupby(['account_no'])['txn_amount'].min()
    txn_am_median = df.groupby(['account_no'])['txn_amount'].median()
    txn_am_count = df.groupby(['account_no'])['txn_amount'].count()
    txn_am_sum = df.groupby(['account_no'])['txn_amount'].sum()

    df_txn = pd.concat([txn_am_mean, txn_am_std, txn_am_max, txn_am_min, txn_am_median, txn_am_count, txn_am_sum], axis=1, join='inner')
    df_txn.columns = ['txn_mean', 'txn_std', 'txn_max', 'txn_min', 'txn_median', 'txn_count', 'txn_sum']
    df_txn = df_txn.fillna(0)

    cols = ['CR', 'DR', 'h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'weekend', 'weekday', 'holidays']
    df_sum = list()
    for col in cols:
        if not len(df_sum) == 0:
            df_sum = pd.concat([df_sum, df.groupby(['account_no'])[col].sum()], axis=1, join='inner')
        else:
            df_sum = df.groupby(['account_no'])[col].sum()

    df_sum.columns = cols

    df_date = df.groupby(['account_no'])['date'].apply(lambda x: x.sort_values().drop_duplicates().diff().fillna(0).median())
    df_date = df_date.dt.days

    df_txn = pd.concat([df_txn, df_sum, df_date], axis=1, join='inner')

    # Preprocess Credit transaction
    whole_sale = {2741, 2791, 2842, 5013, 5021, 5039, 5044, 5045, 5046, 5047, 5051, 5065, 5072, 5074, 5085, 5094,
                  5099, 5111, 5122, 5131, 5137, 5139, 5169, 5172, 5192, 5193, 5198, 5199, 7375, 7379, 7829, 8734}
    contract_services = {742,763,780,1520,1711,1731,1740,1750,1761,1771,1799}
    airlines = set(range(3000,3300)) | {4511}
    rental_car = set(range(3351, 3442)) | {7512}
    hotel = set(range(3501, 3787)) | {7011}
    transport = set(range(4011, 4790))
    utilities = set(range(4812, 4817)) | {4821} | set(range(4899, 4901))
    retail = set(range(5200,5500))
    automobile = set(range(5511, 5600))
    clothing = set(range(5611, 5700))
    misc = set(range(5712, 6000))
    quasi_cash = {4829,6050,6051,6529,6530,6534}
    service_providers = {6010,6011,6012,6531,6532,6533,6211,6300,6381,6399,7011,7012,7032,7033}
    personal = set(range(7210, 7300))
    business = set(range(7311, 7524))
    repair = set(range(7531, 7700))
    entertain = set(range(7829, 8000))
    prefessional = set(range(8011, 9000))
    government = set(range(9211, 9951))

    mcc = {
    'whole_sale': whole_sale, 
    'contract_services': contract_services,
    'airlines': airlines,
    'rental_car': rental_car,
    'hotel': hotel,
    'transport': transport,
    'utilities':utilities,
    'retail': retail,
    'automobile': automobile,
    'clothing': clothing,
    'misc': misc,
    'quasi_cash': quasi_cash,
    'service_providers': service_providers,
    'personal': personal,
    'business': business,
    'repair': repair,
    'entertain': entertain,
    'prefessional': prefessional,
    'government': government,
    }

    for k, v in mcc.items():
        df_cred[k] = df_cred['mer_cat_code'].isin(v).astype(int)

    txn_cr_mean = df_cred.groupby(['card_no'])['txn_amount'].mean()
    txn_cr_std = df_cred.groupby(['card_no'])['txn_amount'].std()
    txn_cr_max = df_cred.groupby(['card_no'])['txn_amount'].max()
    txn_cr_min = df_cred.groupby(['card_no'])['txn_amount'].min()
    txn_cr_median = df_cred.groupby(['card_no'])['txn_amount'].median()
    txn_cr_count = df_cred.groupby(['card_no'])['txn_amount'].count()
    txn_cr_sum = df_cred.groupby(['card_no'])['txn_amount'].sum()

    df_txn_cr = pd.concat([txn_cr_mean, txn_cr_std, txn_cr_max, txn_cr_min, txn_cr_median, txn_cr_count, txn_cr_sum], axis=1, join='inner')
    df_txn_cr.columns = ['txn_cr_mean', 'txn_cr_std', 'txn_cr_max', 'txn_cr_min', 'txn_cr_median', 'txn_cr_count', 'txn_cr_sum']
    df_txn_cr = df_txn_cr.fillna(0)

    df_cr_hr = df_cred.groupby(['card_no'])['txn_hour'].median()
    df_mer_cat = df_cred.drop(['txn_date', 'txn_hour', 'txn_amount', 'mer_cat_code', 'mer_id'], axis=1)
    df_mer_cat = df_mer_cat.groupby(['card_no']).sum()
    df_cr = pd.concat([df_txn_cr, df_cr_hr, df_mer_cat], axis=1, join='inner')

    # Merge account transaction and credit transaction data
    df_txn = df_txn.reset_index(drop=False)
    df_result = pd.merge(df_acc_x, df_txn, on='account_no', how='left', left_index=True)
    df_cr = df_cr.reset_index(drop=False)
    df_result = pd.merge(df_result, df_cr, on='card_no', how='left', left_index=True)
    df_result = df_result.fillna(0)
    
    drop_cols = ['quasi_cash', 'rental_car', 'contract_services', 'repair', 'personal', 'service_providers', 
                 'hotel', 'whole_sale', 'government', 'txn_cr_median', 'airlines', 'prefessional', 'utilities', 
                 'clothing', 'transport', 'retail', 'txn_cr_mean','txn_hour', 'txn_cr_std', 'h1', 'txn_cr_min', 
                 'txn_cr_max']    
    df_result = df_result.drop(drop_cols, axis=1)
        
    # Merge with Train and test data
    X = pd.merge(df_train, df_result, on='account_no')
    X = X.drop(['account_no', 'label', 'card_no'], axis=1)
    assert len(X) == len(df_train)
    
    col_name = X.columns

    y = df_train['label']
    assert len(y) == len(df_train)

    X_test = pd.merge(df_test, df_result, on='account_no')
    X_test = X_test.drop(['account_no', 'card_no'], axis=1)
    assert len(X_test) == len(df_test)
    
    X_all = pd.concat([X, X_test])
    X_all = StandardScaler().fit_transform(X_all.values)
    
    X = X_all[0:len(df_train)]
    X_test = X_all[len(df_train):]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
            
    label_list = [0, 1]
        
    return X_train, y_train, X_val, y_val, X_test, label_list, col_name


def get_data():
    
    X_train_file = 'data/X_train.csv'
    y_train_file = 'data/y_train.csv'
    X_val_file = 'data/X_val.csv'
    y_val_file = 'data/y_val.csv'
    X_test_file = 'data/X_test.csv'
    label_file = 'data/label_list.csv'
    col_file = 'data/col_name.csv'
    
    if not os.path.exists(X_train_file):
        X_train, y_train, X_val, y_val, X_test, label_list, col_name = load_data()    
        
        # Generate file from array
        gen_file(X_train, X_train_file)
        gen_file(y_train, y_train_file)
        gen_file(X_val, X_val_file)
        gen_file(y_val, y_val_file)
        gen_file(X_test, X_test_file)
        gen_file(label_list, label_file)
        gen_file(col_name, col_file)
        
    else:
        # Read array from file
        X_train = get_array(X_train_file)
        
        y_train = get_array(y_train_file)
        y_train = y_train.flatten()
        
        X_val = get_array(X_val_file)
        
        y_val = get_array(y_val_file)
        y_val = y_val.flatten()
        
        X_test = get_array(X_test_file)
        
        label_list = get_array(label_file)
        label_list = label_list.transpose()
        label_list = list(label_list[0])
    
    return X_train, X_val, y_train, y_val, X_test, label_list


def gen_file(array, filename):
    df = pd.DataFrame(array)
    df.to_csv(filename, header=None, index=False)


def get_array(filename):
    df = pd.read_csv(filename, header=None)
    return np.array(df)

