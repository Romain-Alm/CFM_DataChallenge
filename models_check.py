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
from tensorflow.keras.layers import (Input, Dense, Embedding, GRU, concatenate, 
                                   Dropout, BatchNormalization, Bidirectional,
                                   Activation, multiply)


def train_model(model_function, X, Y, epochs=50, batch_size=32):
    """
    Fonction générique d'entraînement d'un modèle
    """
    
    # Préparation des données
    train_data,val_data,train_labels,val_labels=split_data(X,Y)
    train_numeric, train_order, train_venue, train_action, train_side, train_trade = prepare_data_pipeline(train_data)
    val_numeric, val_order, val_venue, val_action, val_side, val_trade = prepare_data_pipeline(val_data)
    print("\nRanges des valeurs dans les données :")
    print(f"train_order min: {np.min(train_order)}, max: {np.max(train_order)}")
    print(f"train_venue min: {np.min(train_venue)}, max: {np.max(train_venue)}")
    print(f"train_action min: {np.min(train_action)}, max: {np.max(train_action)}")
    print(f"train_side min: {np.min(train_side)}, max: {np.max(train_side)}")
    print(f"train_trade min: {np.min(train_trade)}, max: {np.max(train_trade)}")

    print("\nValeurs uniques :")
    print(f"train_side unique values: {np.unique(train_side)}")
    print(f"train_trade unique values: {np.unique(train_trade)}")
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
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    return model, history


def train_model2(model_function, epochs=50, batch_size=32):
    """
    Fonction générique d'entraînement d'un modèle
    """
    
    # Préparation des données
    train_data=pd.read_csv("/Users/josealmeida/Desktop/Data Challenge/x_train_split.csv")
    val_data=pd.read_csv("/Users/josealmeida/Desktop/Data Challenge/x_cv_split.csv")
    train_labels=pd.read_csv("/Users/josealmeida/Desktop/Data Challenge/y_train_split.csv")
    val_labels=pd.read_csv("/Users/josealmeida/Desktop/Data Challenge/y_cv_split.csv")
    train_numeric, train_order, train_venue, train_action, train_side, train_trade = prepare_data_pipeline(train_data)
    val_numeric, val_order, val_venue, val_action, val_side, val_trade = prepare_data_pipeline(val_data)
    print("\nRanges des valeurs dans les données :")
    print(f"train_order min: {np.min(train_order)}, max: {np.max(train_order)}")
    print(f"train_venue min: {np.min(train_venue)}, max: {np.max(train_venue)}")
    print(f"train_action min: {np.min(train_action)}, max: {np.max(train_action)}")
    print(f"train_side min: {np.min(train_side)}, max: {np.max(train_side)}")
    print(f"train_trade min: {np.min(train_trade)}, max: {np.max(train_trade)}")

    print("\nValeurs uniques :")
    print(f"train_side unique values: {np.unique(train_side)}")
    print(f"train_trade unique values: {np.unique(train_trade)}")
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
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
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
                      action_dim=3, side_dim=2, trade_dim=2, learning_rate=3e-3):
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

def create_cnn_rnn_model2():
    # Inputs (comme avant)
    numeric_input = Input(shape=(100, 11))
    order_id_input = Input(shape=(100,))
    action_input = Input(shape=(100,))
    side_input = Input(shape=(100,))
    venue_input = Input(shape=(100,))
    trade_input = Input(shape=(100,))

    # Embeddings plus larges
    order_id_embedding = Embedding(input_dim=100, output_dim=16)(order_id_input)
    action_embedding = Embedding(input_dim=3, output_dim=8)(action_input)
    side_embedding = Embedding(input_dim=2, output_dim=4)(side_input)
    venue_embedding = Embedding(input_dim=6, output_dim=8)(venue_input)
    trade_embedding = Embedding(input_dim=2, output_dim=4)(trade_input)

    # Fusion des embeddings
    concatenated_embeddings = concatenate([
        order_id_embedding, action_embedding, side_embedding,
        venue_embedding, trade_embedding
    ], axis=-1)

    # Fusion avec données numériques
    merged = concatenate([numeric_input, concatenated_embeddings], axis=-1)

    # CNN plus profond
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(merged)
    x = Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # GRU bidirectionnelles plus larges
    x = Bidirectional(GRU(128, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(GRU(64))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense layers avec skip connection
    dense1 = Dense(128, activation='relu')(x)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)

    # Sortie
    output = Dense(24, activation='softmax')(dense2)

    model = Model(inputs=[numeric_input, order_id_input, venue_input, 
                         action_input, side_input, trade_input], 
                 outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Learning rate plus faible
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def hybrid_gru_model(input_shape=(100, 11), order_id_dim=100, venue_dim=6, 
                    action_dim=3, side_dim=2, trade_dim=2, learning_rate=1e-3):
    """
    Version stabilisée du modèle GRU hybride
    """
    # Entrées numériques
    numeric_input = Input(shape=input_shape)
    
    # Embeddings avec régularisation augmentée
    order_id_input = Input(shape=(100,))
    order_id_embedding = Embedding(input_dim=order_id_dim, output_dim=12,
                                 embeddings_regularizer=l2(1e-4))(order_id_input)
    
    action_input = Input(shape=(100,))
    action_embedding = Embedding(input_dim=action_dim, output_dim=6,
                               embeddings_regularizer=l2(1e-4))(action_input)
    
    side_input = Input(shape=(100,))
    side_embedding = Embedding(input_dim=side_dim, output_dim=3,
                             embeddings_regularizer=l2(1e-4))(side_input)
    
    venue_input = Input(shape=(100,))
    venue_embedding = Embedding(input_dim=venue_dim, output_dim=6,
                              embeddings_regularizer=l2(1e-4))(venue_input)
    
    trade_input = Input(shape=(100,))
    trade_embedding = Embedding(input_dim=trade_dim, output_dim=3,
                              embeddings_regularizer=l2(1e-4))(trade_input)

    # Concaténation des embeddings avec dropout
    concatenated_embeddings = concatenate([
        order_id_embedding, action_embedding, side_embedding, 
        venue_embedding, trade_embedding
    ], axis=-1)
    concatenated_embeddings = Dropout(0.2)(concatenated_embeddings)
    
    # Prétraitement des données numériques avec régularisation
    numeric_processed = Dense(32, activation='selu',
                            kernel_regularizer=l2(1e-4))(numeric_input)
    numeric_processed = BatchNormalization()(numeric_processed)
    numeric_processed = Dropout(0.2)(numeric_processed)
    
    # Fusion des données
    merged = concatenate([numeric_processed, concatenated_embeddings], axis=-1)
    
    # GRU avec régularisation
    gru_out = Bidirectional(GRU(96, return_sequences=True,
                               kernel_regularizer=l2(1e-4),
                               recurrent_regularizer=l2(1e-4)))(merged)
    gru_out = BatchNormalization()(gru_out)
    gru_out = Dropout(0.3)(gru_out)
    
    # Attention avec régularisation
    attention = Dense(1, use_bias=False,
                     kernel_regularizer=l2(1e-4))(gru_out)
    attention_weights = Activation('softmax')(attention)
    context_vector = multiply([gru_out, attention_weights])
    
    # Global pooling
    x = GlobalAveragePooling1D()(context_vector)
    
    # Couches denses avec régularisation accrue
    x = Dense(128, activation='selu',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='selu',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    outputs = Dense(24, activation='softmax')(x)
    
    # Construction du modèle
    model = Model(
        inputs=[numeric_input, order_id_input, venue_input,
                action_input, side_input, trade_input],
        outputs=outputs
    )
    
    # Compilation avec clipnorm global sur l'optimiseur uniquement
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def simple_lstm_model(input_shape=(100, 11), order_id_dim=100, action_dim=3, side_dim=2, venue_dim=6, trade_dim=2, learning_rate=0.001):
    # Main inputs
    numeric_input = Input(shape=input_shape)
    order_input = Input(shape=(100,))
    venue_input = Input(shape=(100,))
    action_input = Input(shape=(100,))
    side_input = Input(shape=(100,))
    trade_input = Input(shape=(100,))
    
    # Embeddings simples
    order_embedding = Embedding(input_dim=100, output_dim=16)(order_input)
    venue_embedding = Embedding(input_dim=6, output_dim=8)(venue_input)
    action_embedding = Embedding(input_dim=3, output_dim=8)(action_input)
    side_embedding = Embedding(input_dim=2, output_dim=4)(side_input)
    trade_embedding = Embedding(input_dim=2, output_dim=4)(trade_input)

    # Concat embeddings
    concat_embeddings = concatenate([
        order_embedding, venue_embedding, action_embedding, 
        side_embedding, trade_embedding
    ])
    
    # Fusion avec données numériques
    x = concatenate([numeric_input, concat_embeddings], axis=-1)
    
    # Initial processing
    x = Dense(256, activation='selu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # LSTM blocks avec skip connections
    lstm_units = 128
    lstm_out1 = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out1 = LayerNormalization()(lstm_out1)
    lstm_out1 = Dropout(0.2)(lstm_out1)
    
    lstm_out2 = Bidirectional(LSTM(lstm_units, return_sequences=False))(lstm_out1)  # return_sequences=False
    lstm_out2 = LayerNormalization()(lstm_out2)
    lstm_out2 = Dropout(0.2)(lstm_out2)
    
    # Dense finale
    x = Dense(128, activation='selu')(lstm_out2)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(24, activation='softmax')(x)
    
    model = Model(
        inputs=[numeric_input, order_input, venue_input, action_input, side_input, trade_input],
        outputs=outputs
    )
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model