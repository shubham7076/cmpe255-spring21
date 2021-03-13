import pandas as pd
import numpy as np
#import seaborn as sns

#from matplotlib import pyplot as plt

def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)

def prepare_X(df):

    #base list with five features from the dataset
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

    df = df.copy()

    #Adding the dataframe attributes to the features variable
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    return df_num.fillna(0)

class CarPrice:

    def __init__(self):
        file = 'lab2/data/data.csv'
        self.df = pd.read_csv(file,sep=',')
        print(f'{len(self.df)} lines loaded')

        np.random.seed(4)

        n = len(self.df)

        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train = df_shuffled.iloc[:n_train].copy()
        self.df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        self.df_test = df_shuffled.iloc[n_train+n_val:].copy()

        self.y_train = self.df_train[["msrp"]].applymap(np.log1p)
        self.y_val = self.df_val[["msrp"]].applymap(np.log1p)
        self.y_test = self.df_test[["msrp"]].applymap(np.log1p)

        del self.df_train['msrp']
        del self.df_val['msrp']
        del self.df_test['msrp']

    #def validate(self):
        #pass

    @staticmethod
    def linear_regression_reg(X, y, r=0.0):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg

        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def reg_train_data(self,r):
        X_train = prepare_X(self.df_train)
        return self.linear_regression_reg(X_train,self.y_train,r)
    
    def reg_validate_data(self,w_0,w):
        X_val = prepare_X(self.df_val)
        #y_pred =  w_0 + X_val.dot(w)
        return  w_0 + X_val.dot(w)

    def reg_test_data(self,w_0,w):
        X_test = prepare_X(self.df_test)
        return w_0 + X_test.dot(w)

    
def rmse_after_training_and_validation(r):
    w_0, w = car_price.reg_train_data(r)
    rmse_value = rmse(car_price.y_val.to_numpy(),
     car_price.reg_validate_data(w_0, w).to_numpy())
    return rmse_value

def reverse_log_value(df):
    return df.head().applymap(np.expm1)

if __name__ == '__main__':
    car_price = CarPrice()

    # r values to determine min rmse value after training and validation.
    min_rmse_value = min([0, 0.001, 0.01, 0.1, 1, 10], key= rmse_after_training_and_validation)
    print(f"r which produces min rmse on validation data: {min_rmse_value}")

    # Train the data on min rmse value.
    w_0, w = car_price.reg_train_data(min_rmse_value)

    # Testing the values in w_0 & w.
    y_test = car_price.y_test.head()
    y_pred = car_price.reg_test_data(w_0, w).head()

    # Output 

    result = car_price.df_test[["engine_cylinders", "transmission_type", "driven_wheels", "number_of_doors", "market_category", "vehicle_size", "vehicle_style", "highway_mpg", "city_mpg", "popularity"]].head()
    result["msrp"] = reverse_log_value(y_test)
    result["msrp_pred"] = reverse_log_value(y_pred)
    print(result)