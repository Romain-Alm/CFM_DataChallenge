import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, concatenate, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from Data_Shaping import *
from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, concatenate, Dropout, BatchNormalization, Embedding, MultiHeadAttention, GlobalAveragePooling1D, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def train_model(model_function, X, Y, epochs=50, batch_size=32):
    """
    Fonction générique d'entraînement d'un modèle
    """
    
    # Préparation des données
    train_data,val_data,train_labels,val_labels=split_data(X,Y)
    train_numeric, train_order, train_venue, train_action, train_side, train_trade = prepare_data_pipeline(train_data)
    val_numeric, val_order, val_venue, val_action, val_side, val_trade = prepare_data_pipeline(val_data)
    print("Dimensions des données d'entrée:")
    print(f"train_numeric shape: {train_numeric.shape}")
    print(f"train_order unique values: {np.unique(train_order)}")
    print(f"train_venue unique values: {np.unique(train_venue)}")
    print(f"train_action unique values: {np.unique(train_action)}")
    print(f"train_side unique values: {np.unique(train_side)}")
    print(f"train_trade unique values: {np.unique(train_trade)}")
    
    print("\nRanges des valeurs dans les données :")
    print(f"train_order min: {np.min(train_order)}, max: {np.max(train_order)}")
    print(f"train_venue min: {np.min(train_venue)}, max: {np.max(train_venue)}")
    print(f"train_action min: {np.min(train_action)}, max: {np.max(train_action)}")
    print(f"train_side min: {np.min(train_side)}, max: {np.max(train_side)}")
    print(f"train_trade min: {np.min(train_trade)}, max: {np.max(train_trade)}")

    print("\nVérification des valeurs hors limites :")
    print(f"train_order values > 99: {np.sum(train_order > 99)}")
    print(f"train_venue values > 5: {np.sum(train_venue > 5)}")
    print(f"train_action values > 2: {np.sum(train_action > 2)}")
    print(f"train_side values > 1: {np.sum(train_side > 1)}")
    print(f"train_trade values > 1: {np.sum(train_trade > 1)}")
    # Construction du modèle
    model = model_function()
    
    # Entraînement
    history = model.fit(
        [train_numeric, train_order, train_venue, train_action, train_side, train_trade],
        train_labels,
        validation_data=([val_numeric, val_order, val_venue, val_action, val_side, val_trade], 
                        val_labels),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
    )
    
    return model, history



def complexed_gru_reshaped2(input_shape, order_id_dim, action_dim, side_dim, venue_dim, trade_dim, learning_rate=3e-3):
    # Input layer
    inputs = Input(shape=input_shape)  # Cela reste (100, n_features)
    print(f"Shape of inputs: {inputs.shape}")

    # Embedding layers pour les variables catégorielles
    order_id_input = Input(shape=(100,))
    order_id_embedding = Embedding(input_dim=order_id_dim, output_dim=8)(order_id_input)

    action_input = Input(shape=(100,))
    action_embedding = Embedding(input_dim=action_dim, output_dim=4)(action_input)

    side_input = Input(shape=(100,))
    side_embedding = Embedding(input_dim=side_dim, output_dim=4)(side_input)

    venue_input = Input(shape=(100,))
    venue_embedding = Embedding(input_dim=venue_dim, output_dim=4)(venue_input)

    trade_input = Input(shape=(100,))
    trade_embedding = Embedding(input_dim=trade_dim, output_dim=4)(trade_input)

# Concatenation des embeddings directement
    concatenated_embeddings = concatenate([
    order_id_embedding, action_embedding, side_embedding, venue_embedding, trade_embedding
    ], axis=-1)
    # Concaténation des embeddings et des données séquentielles
    #concatenated_embeddings = concatenate([order_id_embedding, action_embedding, side_embedding, venue_embedding, trade_embedding])
    #concatenated_embeddings = RepeatVector(input_shape[0])(concatenated_embeddings)  # Répéter les embeddings pour chaque timestep
    print(f"Shape after concatenation and repeat: {concatenated_embeddings.shape}")
    print(f"Shape of concatenated_embeddings: {concatenated_embeddings.shape}")
    # Combine les embeddings répétés avec les données séquentielles
    x = concatenate([inputs, concatenated_embeddings])
    print(f"Shape after concatenation: {x.shape}")
    print(f"Shape of inputs: {inputs.shape}")
    print(f"Shape of concatenated_embeddings: {concatenated_embeddings.shape}")
    
    # Applique une couche Dense pour ajuster la forme avant les GRU (optionnel)
    x = Dense(128, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='selu')(x)


    # Couches GRU bidirectionnelles
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Couches denses finales
    x = Dense(128, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Sortie
    outputs = Dense(24, activation='softmax')(x)

    # Modèle final
    model = Model(inputs=[inputs, order_id_input, action_input, side_input, venue_input, trade_input], outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


### pour le code 1) finir la partie shaping (dont normalisation + le split train cv) 2) choisir 3/4 modèles différents 
# 3) les faire tourner avec reduceLRonPlatau et EarlyStoppingsur 15 epochs4) faire tourner le plus performant


# def evaluate_models(X, Y, model_functions):
#     """
#     Compare plusieurs modèles sur le même jeu de données
    
#     Parameters:
#     -----------
#     X : DataFrame
#         Données d'entrée complètes
#     Y : array
#         Labels correspondants
#     model_functions : dict
#         Dictionnaire des fonctions de création de modèles à comparer
        
#     Returns:
#     --------
#     DataFrame
#         Résultats de performance pour chaque modèle
#     """
#     results = []
    
#     for model_name, model_func in model_functions.items():
#         # Entraînement du modèle
#         model, history = train_model(model_func, X, Y)
        
#         # Préparation des données de validation pour l'évaluation finale
#         _, val_data, _, val_labels = split_data(X, Y)
#         val_numeric, val_order, val_venue, val_action, val_side, val_trade = prepare_data_pipeline(val_data)
        
#         # Évaluation sur les données de validation
#         val_score = model.evaluate(
#             [val_numeric, val_order, val_venue, val_action, val_side, val_trade],
#             val_labels,
#             verbose=0
#         )
        
#         # Enregistrement des résultats
#         results.append({
#             'model_name': model_name,
#             'validation_loss': val_score[0],
#             'validation_accuracy': val_score[1],
#             'best_epoch': len(history.history['loss']),
#             'min_val_loss': min(history.history['val_loss']),
#             'max_val_accuracy': max(history.history['val_accuracy'])
#         })
    
#     return pd.DataFrame(results)

# def run_model_comparison(X, Y):
#     """
#     Fonction principale pour exécuter la comparaison des modèles
    
#     Parameters:
#     -----------
#     X : DataFrame
#         Données d'entrée
#     Y : array
#         Labels correspondants
        
#     Returns:
#     --------
#     DataFrame
#         Résultats comparatifs des différents modèles
#     """
#     model_functions = {
#         'lstm': create_lstm_model,
#         'gru': create_gru_model,
#         'transformer': create_transformer_model
#         # Ajoutez d'autres modèles ici selon besoin
#     }
    
#     results = evaluate_models(X, Y, model_functions)
    
#     print("\nRésultats de la comparaison des modèles :")
#     print(results[['model_name', 'validation_accuracy', 'validation_loss', 'best_epoch']])
    
#     return results