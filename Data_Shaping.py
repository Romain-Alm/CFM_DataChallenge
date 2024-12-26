import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

### 1 Preparation and data analysis
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






### 2 feature engineering






### 3 model preparation






