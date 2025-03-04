import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, concatenate, Dropout, BatchNormalization, Bidirectional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, concatenate, Dropout, BatchNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


### 1 data analysis
def analyze_basic_stats(df):
    """
    Analyse statistique basique des colonnes numériques
    """
    # Statistiques descriptives
    numeric_stats = df.describe()[['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux']]
    
    # Calcul des skewness et kurtosis pour détecter les distributions anormales
    skewness = df.select_dtypes(include=[np.number]).skew()
    kurtosis = df.select_dtypes(include=[np.number]).kurtosis()
    
    return numeric_stats, skewness, kurtosis

def analyze_categorical_features(df):
    """
    Analyse des variables catégorielles
    """
    categorical_cols = ['venue', 'action', 'side', 'trade']
    distribution = {col: df[col].value_counts(normalize=True) for col in categorical_cols}
    
    # Analyse des transitions d'actions (A->D, A->U, etc.)
    action_transitions = df.groupby('obs_id')['action'].apply(lambda x: pd.Series(zip(x, x.shift(-1)))).value_counts()
    
    return distribution, action_transitions

def analyze_temporal_patterns(sequences):
    """
    Analyse des patterns temporels dans les séquences
    """
    # Analyse de la distribution des flux par séquence
    sequence_stats = sequences.agg({
        'flux': ['mean', 'std', 'min', 'max'],
        'trade': 'sum',  # Nombre de trades par séquence
        'action': lambda x: x.value_counts()  # Distribution des actions par séquence
    })
    
    return sequence_stats

def analyze_price_dynamics(df):
    """
    Analyse de la dynamique des prix
    """
    # Spread moyen
    df['spread'] = df['ask'] - df['bid']
    
    # Ratio de déséquilibre du carnet d'ordres
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
    
    price_dynamics = {
        'spread_stats': df['spread'].describe(),
        'imbalance_stats': df['order_imbalance'].describe()
    }
    
    return price_dynamics

def check_data_quality(df):
    """
    Vérification de la qualité des données
    """
    # Valeurs manquantes
    missing_values = df.isnull().sum() ### after verification, no missing values
    
    # Vérifications de cohérence
    consistency_checks = {
        'ask_greater_than_bid': (df['ask'] >= df['bid']).all(),
        'positive_sizes': (df[['bid_size', 'ask_size']] >= 0).all().all(),
        'valid_actions': df['action'].isin(['A', 'D', 'U']).all()
    }
    
    return missing_values, consistency_checks

def generate_exploration_report(df, sequences):
    """
    Génère un rapport complet d'exploration
    """
    basic_stats, skewness, kurtosis = analyze_basic_stats(df)
    cat_dist, action_trans = analyze_categorical_features(df)
    temporal_patterns = analyze_temporal_patterns(sequences)
    price_dynamics = analyze_price_dynamics(df)
    missing_vals, consistency = check_data_quality(df)
    
    return {
        'basic_stats': basic_stats,
        'distributions': {'skewness': skewness, 'kurtosis': kurtosis},
        'categorical_analysis': cat_dist,
        'action_transitions': action_trans,
        'temporal_patterns': temporal_patterns,
        'price_dynamics': price_dynamics,
        'data_quality': {'missing': missing_vals, 'consistency': consistency}
    }





def create_sequence_features(sequence):
    """
    Crée des features pour une séquence donnée
    """
    sequence = sequence.copy()
    
    # Features basiques
    sequence['spread'] = sequence['ask'] - sequence['bid']
    sequence['order_imbalance'] = (sequence['bid_size'] - sequence['ask_size']) / (sequence['bid_size'] + sequence['ask_size'])
    
    # Features cumulatives
    sequence['cumul_volume'] = sequence['flux'].cumsum()
    sequence['price_change']=sequence['price'] - sequence['price'].iloc[0]

    ### la position dans le carnet d'oredre est donné par order_id
    
    ### anomalies detection
    sequence['price_anomaly'] = (sequence['ask'] < sequence['bid']).astype(int) ### catégorise dès que bid>ask
    
    return sequence

def create_sequence_features2(sequence):
    sequence = sequence.copy()
    
    # Mid price (très informatif pour la classification)
    sequence['mid_price'] = (sequence['ask'] + sequence['bid']) / 2
    sequence['order_imbalance'] = (sequence['bid_size'] - sequence['ask_size']) / (sequence['bid_size'] + sequence['ask_size'])

    # Order flow imbalance amélioré
    sequence['price_change']=sequence['price'] - sequence['price'].iloc[0]
    sequence['cumul_volume'] = sequence['flux'].cumsum()

    # Relative spread (meilleur que le spread absolu)
    sequence['relative_spread'] = (sequence['ask'] - sequence['bid']) 
    # Transformations log avec offsets spécifiques
    sequence['log_ask_size'] = np.log(sequence['ask_size'] + 1)  # ask_size est toujours positif
    sequence['log_bid_size'] = np.log(sequence['bid_size'] + 3)  # offset de 3 pour bid_size qui peut être négatif
    sequence['log_flux'] = np.log(sequence['flux'] + 50001) 
    return sequence



def create_sequence_features3(sequence):
    sequence = sequence.copy()
    
    # Handle extreme values with log transform
    eps = 1e-8
    sequence['log_price'] = np.sign(sequence['price']) * np.log1p(np.abs(sequence['price']))
    sequence['log_bid_size'] = np.log1p(sequence['bid_size'] + eps)
    sequence['log_ask_size'] = np.log1p(sequence['ask_size'] + eps)
    sequence['log_flux'] = np.sign(sequence['flux']) * np.log1p(np.abs(sequence['flux']))
    sequence['log_bid'] = np.sign(sequence['bid']) * np.log1p(np.abs(sequence['bid']))
    sequence['log_ask'] = np.sign(sequence['ask']) * np.log1p(np.abs(sequence['ask']))

    
    # Relative features (less affected by scale)
    
    sequence['mid_price'] = (sequence['ask'] + sequence['bid']) / 2
    sequence['price_change'] = sequence['price'] - sequence['price'].iloc[0]
    sequence['relative_spread'] = (sequence['ask'] - sequence['bid']) / ((sequence['ask'] + sequence['bid'])/2)
    sequence['order_imbalance'] = (sequence['bid_size'] - sequence['ask_size']) / (sequence['bid_size'] + sequence['ask_size'])
    
    # Accumulative features
    sequence['cumul_volume'] = sequence['flux'].cumsum()
    
    return sequence   
def process_sequences(df):
    """
    Fonction principale de traitement des séquences
    """
    # Création des features pour chaque séquence
    sequences_with_features = pd.concat([
        create_sequence_features(sequence) 
        for _, sequence in df.groupby('obs_id')
    ])
    
    return sequences_with_features

def process_sequences2(df):
    """
    Fonction principale de traitement des séquences
    """
    # Création des features pour chaque séquence
    sequences_with_features = pd.concat([
        create_sequence_features2(sequence) 
        for _, sequence in df.groupby('obs_id')
    ])
    
    return sequences_with_features

def process_sequences3(df):
    """
    Fonction principale de traitement des séquences
    """
    # Création des features pour chaque séquence
    sequences_with_features = pd.concat([
        create_sequence_features3(sequence) 
        for _, sequence in df.groupby('obs_id')
    ])
    
    return sequences_with_features

def encoding_columns(df):
    df['action_encoded']=df['action'].map({'A': 0, 'D': 1, 'U': 2})
    df['side_encoded']=df['side'].map({'A': 0, 'B': 1})
    df['trade_encoded']=df['trade'].map({True: 1, False: 0})
    df = df.drop(columns=['action','side','trade'])
    return df

def split_data(X,Y):
    Y=Y.iloc[:,1]
    x_train=X.head(int(0.8 * len(X)))
    x_cv=X.tail(int(0.2*len(X)))
    y_train=Y.head(int(0.8*len(Y)))
    y_cv=Y.tail(int(0.2*len(Y)))

    return x_train, x_cv, y_train, y_cv

def prepare_data_pipeline3(df):
    """
    Pipeline principal de préparation des données retournant les features numériques 
    et catégorielles sous forme de numpy arrays
    """
    # Application du feature engineering
    df = process_sequences3(df)
    print("Colonnes disponibles avant encodage:", df.columns.tolist())
    # Encodage des colonnes catégorielles
    df_encoded = encoding_columns(df)
    print("Colonnes disponibles après encodage:", df_encoded.columns.tolist())
    
    # Préparation des features numériques
    numeric_features = ['log_price', 'log_bid', 'log_ask', 'log_bid_size', 'log_ask_size', 'log_flux',
                       'relative_spread', 'order_imbalance', 'cumul_volume', 'price_change',
                       'mid_price']
    
    # Normalisation et reshape des features numériques
    scaler = StandardScaler()
    numeric_array = scaler.fit_transform(df[numeric_features].values)
    numeric_array = numeric_array.reshape(-1, 100, len(numeric_features))
    print("successfull normalizayion")
    
    # Préparation des arrays catégoriels
    order_ids = df['order_id'].values.reshape(-1, 100)
    venues = df['venue'].values.reshape(-1, 100)
    actions = df_encoded['action_encoded'].values.reshape(-1, 100)
    sides = df_encoded['side_encoded'].values.reshape(-1, 100)
    trades = df_encoded['trade_encoded'].values.reshape(-1, 100)
    
    return numeric_array, order_ids, venues, actions, sides, trades




def prepare_data_pipeline2(df):
    """
    Pipeline principal de préparation des données retournant les features numériques 
    et catégorielles sous forme de numpy arrays
    """
    # Application du feature engineering
    df = process_sequences2(df)
    print("Colonnes disponibles avant encodage:", df.columns.tolist())
    # Encodage des colonnes catégorielles
    df_encoded = encoding_columns(df)
    print("Colonnes disponibles après encodage:", df_encoded.columns.tolist())
    
    # Préparation des features numériques
    numeric_features = ['price', 'bid', 'ask', 'log_bid_size', 'log_ask_size', 'log_flux',
                       'relative_spread', 'order_imbalance', 'cumul_volume', 'price_change',
                       'mid_price']
    
    # Normalisation et reshape des features numériques
    scaler = StandardScaler()
    numeric_array = scaler.fit_transform(df[numeric_features].values)
    numeric_array = numeric_array.reshape(-1, 100, len(numeric_features))
    print("successfull normalizayion")

    # Préparation des arrays catégoriels
    order_ids = df['order_id'].values.reshape(-1, 100)
    venues = df['venue'].values.reshape(-1, 100)
    actions = df_encoded['action_encoded'].values.reshape(-1, 100)
    sides = df_encoded['side_encoded'].values.reshape(-1, 100)
    trades = df_encoded['trade_encoded'].values.reshape(-1, 100)
    
    return numeric_array, order_ids, venues, actions, sides, trades





def prepare_data_pipeline(df):
    """
    Pipeline principal de préparation des données retournant les features numériques 
    et catégorielles sous forme de numpy arrays
    """
    # Application du feature engineering
    df = process_sequences(df)
    print("Colonnes disponibles avant encodage:", df.columns.tolist())
    # Encodage des colonnes catégorielles
    df_encoded = encoding_columns(df)
    print("Colonnes disponibles après encodage:", df_encoded.columns.tolist())
    
    # Préparation des features numériques
    numeric_features = ['price', 'bid', 'ask', 'bid_size', 'ask_size', 'flux',
                       'spread', 'order_imbalance', 'cumul_volume', 'price_change',
                       'price_anomaly']
    
    # Normalisation et reshape des features numériques
    scaler = StandardScaler()
    numeric_array = scaler.fit_transform(df[numeric_features].values)
    numeric_array = numeric_array.reshape(-1, 100, len(numeric_features))
    print("successfull normalizayion")
    
    # Préparation des arrays catégoriels
    order_ids = df['order_id'].values.reshape(-1, 100)
    venues = df['venue'].values.reshape(-1, 100)
    actions = df_encoded['action_encoded'].values.reshape(-1, 100)
    sides = df_encoded['side_encoded'].values.reshape(-1, 100)
    trades = df_encoded['trade_encoded'].values.reshape(-1, 100)
    
    return numeric_array, order_ids, venues, actions, sides, trades


