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
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.75, patience=3)
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


def basic_gru_model(input_shape=(100, 11), order_id_dim=100, venue_dim=6, 
                   action_dim=3, side_dim=2, trade_dim=2, learning_rate=3e-3):
    """
    Modèle GRU simplifié avec embeddings de base
    """
    # Entrées numériques
    numeric_input = Input(shape=input_shape)
    
    # Embeddings simples
    order_id_input = Input(shape=(100,))
    order_id_embedding = Embedding(input_dim=order_id_dim, output_dim=8)(order_id_input)
    
    action_input = Input(shape=(100,))
    action_embedding = Embedding(input_dim=action_dim, output_dim=4)(action_input)
    
    side_input = Input(shape=(100,))
    side_embedding = Embedding(input_dim=side_dim, output_dim=2)(side_input)
    
    venue_input = Input(shape=(100,))
    venue_embedding = Embedding(input_dim=venue_dim, output_dim=3)(venue_input)
    
    trade_input = Input(shape=(100,))
    trade_embedding = Embedding(input_dim=trade_dim, output_dim=2)(trade_input)

    # Concaténation
    concatenated_embeddings = concatenate([
        order_id_embedding, action_embedding, side_embedding, 
        venue_embedding, trade_embedding
    ], axis=-1)
    
    # Fusion avec données numériques
    merged = concatenate([numeric_input, concatenated_embeddings], axis=-1)
    
    # Couches GRU simples
    x = Bidirectional(GRU(64, return_sequences=True))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Bidirectional(GRU(32))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Couches denses finales
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(24, activation='softmax')(x)
    
    # Modèle
    model = Model(
        inputs=[numeric_input, order_id_input, venue_input,
                action_input, side_input, trade_input],
        outputs=outputs
    )
    
    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def advanced_gru_model(input_shape=(100, 11), order_id_dim=100, venue_dim=6, 
                      action_dim=3, side_dim=2, trade_dim=2, learning_rate=1e-3):
    """
    Version avancée du modèle GRU avec améliorations ciblées
    """
    # Entrées numériques
    numeric_input = Input(shape=input_shape)
    
    # Embeddings plus larges pour capturer plus d'informations
    order_id_input = Input(shape=(100,))
    order_id_embedding = Embedding(input_dim=order_id_dim, output_dim=16)(order_id_input)
    
    action_input = Input(shape=(100,))
    action_embedding = Embedding(input_dim=action_dim, output_dim=8)(action_input)
    
    side_input = Input(shape=(100,))
    side_embedding = Embedding(input_dim=side_dim, output_dim=4)(side_input)
    
    venue_input = Input(shape=(100,))
    venue_embedding = Embedding(input_dim=venue_dim, output_dim=8)(venue_input)
    
    trade_input = Input(shape=(100,))
    trade_embedding = Embedding(input_dim=trade_dim, output_dim=4)(trade_input)

    # Traitement séparé des embeddings
    concatenated_embeddings = concatenate([
        order_id_embedding, action_embedding, side_embedding, 
        venue_embedding, trade_embedding
    ], axis=-1)
    
    # Prétraitement des données numériques
    numeric_processed = Dense(32, activation='selu')(numeric_input)
    numeric_processed = BatchNormalization()(numeric_processed)
    
    # Fusion des données
    merged = concatenate([numeric_processed, concatenated_embeddings], axis=-1)
    
    # Premier bloc GRU avec skip connection
    main_path = Bidirectional(GRU(128, return_sequences=True))(merged)
    main_path = BatchNormalization()(main_path)
    main_path = Dropout(0.3)(main_path)
    
    # Couche d'attention simple
    attention = Dense(1, use_bias=False)(main_path)
    attention_weights = Activation('softmax')(attention)
    context_vector = multiply([main_path, attention_weights])
    
    # Deuxième bloc GRU avec taille réduite
    x = Bidirectional(GRU(64, return_sequences=False))(context_vector)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Couches denses avec architecture pyramidale
    x = Dense(256, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, activation='selu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Couche de sortie
    outputs = Dense(24, activation='softmax')(x)
    
    # Construction du modèle
    model = Model(
        inputs=[numeric_input, order_id_input, venue_input,
                action_input, side_input, trade_input],
        outputs=outputs
    )
    
    # Compilation avec paramètres optimisés
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model