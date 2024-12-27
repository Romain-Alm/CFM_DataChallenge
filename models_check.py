import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Embedding, GRU, concatenate, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


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