import numpy as np
import pandas as pd


# mean FA in ABCD
def load_fa_data():
    """### Load Data"""
    raw_fa_df = np.genfromtxt('data/fa.csv', delimiter=',', skip_header = 1)
    base_FA = raw_fa_df[:,:-1]
    y_2classes = raw_fa_df[:,-1]

    num_classes= 2
    return base_FA, y_2classes, num_classes

# eye movement
def load_eye_data():
    raw_eye_df = pd.read_csv('data/eye_movement.csv')
    y = raw_eye_df['label'].to_numpy().astype(np.float32)
    raw_eye_df.drop(['label'],axis=1,inplace=True)
    normalized_raw_eye_df = (raw_eye_df - raw_eye_df.mean())/raw_eye_df.std()
    X = normalized_raw_eye_df.to_numpy().astype(np.float32)
    num_classes = 3

    return X, y, num_classes

# cortical thickness in ADNI
def load_ct_data():
    raw_ct_df = pd.read_excel('data/DT_thickness.xlsx')

    raw_ct_df = raw_ct_df.loc[raw_ct_df['DX'].isin(['CN','EMCI','LMCI','AD'])]
    y = raw_ct_df['DX'].map({'CN':0,'EMCI':1,'LMCI':2,'AD':3}).to_numpy().astype(np.float32)

    # raw_ct_df = raw_ct_df.loc[raw_ct_df['DX'].isin(['CN','EMCI'])]
    # y = raw_ct_df['DX'].map({'CN':0,'EMCI':1}).to_numpy().astype(np.float32)

    raw_ct_df.drop(['Subject','PTID','DX'],axis=1, inplace=True)
    X = raw_ct_df.to_numpy().astype(np.float32)
    num_classes = 4

    return X, y, num_classes

# tau in ADNI
def load_tau_data():
    raw_tau_df = pd.read_excel('data/Tau_SUVR.xlsx')

    raw_tau_df = raw_tau_df.loc[raw_tau_df['DX'].isin(['CN','EMCI','LMCI','AD'])]
    y = raw_tau_df['DX'].map({'CN':0,'EMCI':1,'LMCI':2,'AD':3}).to_numpy().astype(np.float32)

    # raw_tau_df = raw_tau_df.loc[raw_tau_df['DX'].isin(['CN','EMCI'])]
    # y = raw_tau_df['DX'].map({'CN':0,'EMCI':1}).to_numpy().astype(np.float32)
    raw_tau_df.drop(['PTID','DX','PTGENDER','PTETHCAT','PTMARRY','APOE4','EXAMDATE','SCAN','AGE','PTEDUCAT','PTRACCAT'],axis=1, inplace=True)
    X = raw_tau_df.to_numpy().astype(np.float32)
    num_classes = 4

    return X, y, num_classes

# fdg in ADNI
def load_fdg_data():
    raw_fdg_df = pd.read_excel('data/FDG_SUVR.xlsx')

    raw_fdg_df = raw_fdg_df.loc[raw_fdg_df['DX'].isin(['CN','EMCI','LMCI','AD'])]
    y = raw_fdg_df['DX'].map({'CN':0,'EMCI':1,'LMCI':2,'AD':3}).to_numpy().astype(np.float32)

    # raw_fdg_df = raw_fdg_df.loc[raw_fdg_df['DX'].isin(['CN','EMCI'])]
    # y = raw_fdg_df['DX'].map({'CN':0,'EMCI':1}).to_numpy().astype(np.float32)

    raw_fdg_df.drop(['PTID','DX','PTGENDER','PTETHCAT','PTMARRY','APOE4','EXAMDATE','SCAN','AGE','PTEDUCAT','PTRACCAT'],axis=1, inplace=True)
    X = raw_fdg_df.to_numpy().astype(np.float32)
    num_classes = 4

    return X, y, num_classes

# amy in ADNI
def load_amy_data():
    raw_amy_df = pd.read_excel('data/Amyloid_SUVR.xlsx')

    # raw_amy_df = raw_amy_df.loc[raw_amy_df['DX'].isin(['CN','EMCI','LMCI','AD'])]
    # y = raw_amy_df['DX'].map({'CN':0,'EMCI':0,'LMCI':1,'AD':1}).to_numpy().astype(np.float32)
    raw_amy_df = raw_amy_df.loc[raw_amy_df['DX'].isin(['CN','EMCI'])]
    y = raw_amy_df['DX'].map({'CN':0,'EMCI':1}).to_numpy().astype(np.float32)

    raw_amy_df.drop(['PTID','DX','PTGENDER','PTETHCAT','PTMARRY','APOE4','EXAMDATE','SCAN','AGE','PTEDUCAT','PTRACCAT'],axis=1, inplace=True)
    X = raw_amy_df.to_numpy().astype(np.float32)
    num_classes = 2

    return X, y, num_classes

def load_cardio_data():
    raw_cardio_df = pd.read_csv('data/cardio.csv',sep=';')
    y = raw_cardio_df['cardio'].to_numpy().astype(np.float32)
    raw_cardio_df.drop(['id','cardio'],axis=1,inplace=True)
    normalized_raw_eye_df = (raw_cardio_df - raw_cardio_df.mean())/raw_cardio_df.std()
    X = normalized_raw_eye_df.to_numpy().astype(np.float32)
    num_classes = 2

    return X, y, num_classes

def load_forest_data():
    raw_forest_df = pd.read_csv('data/forest.csv',sep=',')
    y = raw_forest_df['Cover_Type'].to_numpy().astype(np.float32)
    y -= 1
    raw_forest_df.drop(['Id','Cover_Type','Soil_Type7','Soil_Type15'],axis=1,inplace=True)

    X = raw_forest_df.to_numpy().astype(np.float32)
    num_classes = 7

    return X, y, num_classes

def load_data(type='fa'):
    if type == 'fa':
        return load_fa_data()
    elif type == 'eye':
        return load_eye_data()
    elif type == 'ct':
        return load_ct_data()
    elif type == 'cardio':
        return load_cardio_data()
    elif type == 'forest':
        return load_forest_data()
    elif type == 'tau':
        return load_tau_data()
    elif type == 'fdg':
        return load_fdg_data()
    elif type == 'amy':
        return load_amy_data()