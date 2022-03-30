import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats




def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

# for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
#                          key= lambda x: -x[1])[:20]:
#     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))




def print_null(df: pd.DataFrame, name="TABLE"):
    '''
    Prints the number of null values in a desired way
    '''
    print(name)
    print(10*"-")
    print(df.isnull().sum())
    print(10*"*"+"\n")




def generalized_hist_v2(bin_index, values, bins_num, stat='sum'):

    '''
    Returns the binning statistics and center of bins for values array

    Parameters:
    ----------
    bin_index: The values on which bin intervals to be set
    values: The array whose binning statistic will be returned
    bins_num: Total number of bins
    stat: Type of binning statistic to be returned

    Returns:
    --------
    bin_centers: Array containing the center of bins
    bin_stats: Array containing binning statistics
    '''

    bin_stats, bin_edges, _ = \
                        stats.binned_statistic(bin_index, values, stat, bins=bins_num)

    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2


    return bin_centers, bin_stats



def split_time_series(df: pd.DataFrame, val_idx, drop_cols_fit: list, start_idx=0):
    '''
    Splits the time  series dataframe, respecting its time component
    I.e. validation will be future of training set
    '''

    # Time component of the data frame
    dates = df['date_block_num']

    # Mask values to get training and validation dataset
    mask_train = (dates<val_idx) & (start_idx <= dates)
    mask_val = dates==val_idx

    # Extract training and validation set
    X_train = df.loc[mask_train].drop(drop_cols_fit, axis=1)
    X_val =  df.loc[mask_val].drop(drop_cols_fit, axis=1)

    # Extract corresponding labels
    y_train = df.loc[mask_train, 'target'].values.flatten()
    y_val =  df.loc[mask_val, 'target'].values.flatten()
    
    return X_train, X_val, y_train, y_val





def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Changes column types in the dataframe:
     
        `float64` type to `float32`
        `int64`   type to `int32`
    '''
    
    # Select columns to downcast
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype == "int64"]
    
    # Downcast
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    
    return df




def is_scalar_nan(x):
    return isinstance(x, numbers.Real) and math.isnan(x)

def _object_dtype_isnan(X):
    return X != X

def _get_dense_mask(X, value_to_mask):
    if is_scalar_nan(value_to_mask):
        if X.dtype.kind == "f":
            Xt = np.isnan(X)
        elif X.dtype.kind in ("i", "u"):
            # can't have NaNs in integer array.
            Xt = np.zeros(X.shape, dtype=bool)
        else:
            # np.isnan does not work on object dtypes.
            Xt = _object_dtype_isnan(X)
    else:
        Xt = X == value_to_mask

    return Xt


def category_distance(X,Y):
    '''
    Calculate the distance between two arrays that consists of 
    categorical values, which are encoded as integer values
    '''
    
    distances = np.zeros(shape=(X.shape[0], Y.shape[0]))
    if (X.size!=0) and (Y.size!=0):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(X)
        sX = enc.transform(X)
        sY = enc.transform(Y)
        distances = euclidean_distances(sX, sY, squared=True)

    return distances


def distance_mixed(X, Y, category_mask, squared=False):
    
    '''
    Calculates pairwise total distance = distance between numerical values + distance between categorical values
    Given two 2d arrays of X, Y
    '''

    # print(f"Inside mixed, Shape of X and Y: {X.shape} and {Y.shape}")

    # Separate real(numerical) and categorical values
    X_cat, X_num = X[:,category_mask], X[:,~category_mask]
    Y_cat, Y_num = Y[:,category_mask], Y[:,~category_mask]
    
    # Calcuate distances between numerical values
    distances = euclidean_distances(X_num, Y_num, squared=True)
    # print(distances)
    
    # Calculate distances between categorical values
    distances += category_distance(X_cat, Y_cat)
    
    return distances



def distance_mixed2(X, Y, category_mask, squared=True):

    '''
    Calculates distance = distance between numerical values + distance between categorical values
    Given two arrays of X, Y
    '''
    distance = np.sum((X[~category_mask] - Y[~category_mask])**2)
    distance += np.sum(X[category_mask] != Y[category_mask])
    
    return distance


    


def nan_mixed_distances(
    X, Y, *, category_mask=None, squared=False, missing_values=np.nan, copy=True):
    if category_mask is None:
        category_mask = np.array([False, True, True] + [True]*6 + [False]*28)
    
    # Get missing mask for X
    missing_X = _get_dense_mask(X, missing_values) & ~category_mask

    # Get missing mask for Y
    missing_Y = (missing_X if Y is X else _get_dense_mask(Y, missing_values)) & ~category_mask

    # set missing values to zero
    X[missing_X] = 0
    Y[missing_Y] = 0
    
    # print(f"Inside nan_mixed, Shape of X and Y: {X.shape} and {Y.shape}")

    distances = distance_mixed(X, Y, category_mask, squared=True)
    

    # Adjust distances for missing values
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)
    
    np.clip(distances, 0, None, out=distances)

    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        np.fill_diagonal(distances, 0.0)
        
        
        
    # Renormalize the distance by the amount of missing values
    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)
    distances[present_count == 0] = np.nan
    # avoid divide by zero
    np.maximum(1, present_count, out=present_count)
    distances /= present_count
    distances *= X.shape[1]
    
    X[missing_X] = missing_values
    Y[missing_Y] = missing_values

    if not squared:
        np.sqrt(distances, out=distances)
    
    return distances






def nan_mixed_distances2(
    X, Y, *, category_mask=None, squared=False, missing_values=np.nan, copy=True
):
    if category_mask is None:
        category_mask = np.array([False, True, True] + [True]*6 + [False]*28)
        
    # Get missing mask for X
    missing_X = _get_dense_mask(X, missing_values) & ~category_mask

    # Get missing mask for Y
    missing_Y = (missing_X if Y is X else _get_dense_mask(Y, missing_values)) & ~category_mask

    # set missing values to zero
    X[missing_X] = 0
    Y[missing_Y] = 0
    
    # print(f"Inside nan_mixed, Shape of X and Y: {X.shape} and {Y.shape}")

    distances = distance_mixed2(X, Y, category_mask, squared=True)
    

    # Adjust distances for missing values
    XX = X * X
    YY = Y * Y
    distances -= np.dot(XX, missing_Y.T)
    distances -= np.dot(missing_X, YY.T)
    
    if distances < 0:
        distances = 0

    if X is Y:
        distances = 0
        
        
        
    # Renormalize the distance by the amount of missing values
    present_X = 1 - missing_X
    present_Y = present_X if Y is X else ~missing_Y
    present_count = np.dot(present_X, present_Y.T)
    
    if present_count == 0:
        distances = np.nan
    # avoid divide by zero
    if present_count != 0:
        distances /= present_count
        distances *= X.shape[0]
    
    X[missing_X] = missing_values
    Y[missing_Y] = missing_values

    if not squared:
        distances = math.sqrt(distances)
    
    return distances




# from sklearn.impute import KNNImputer

# X = np.array([[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]])
# imputer = KNNImputer(n_neighbors=10, metric=nan_mixed_distances2)

# data_sales.drop(columns=['target'], inplace=True)
# imputer.fit(data_sales)
# data_sales.shape
# X_imputed = imputer.transform(data_sales)