from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model,load_model
import tensorflow as tf
import pandas as pd
import numpy as np
#from sklearn.metrics import log_loss

# Define a function to create a base network for each horse
def create_base_network(input_layer):
    nn = Dense(128, activation='relu')(input_layer)
    nn = Dropout(0.4)(nn)
    nn = BatchNormalization()(nn)
    nn = Dense(64, activation='relu')(nn)
    nn = Dropout(0.3)(nn)
    nn = BatchNormalization()(nn)
    nn = Dense(32, activation='relu')(nn)
    nn = Dropout(0.3)(nn)
    nn = BatchNormalization()(nn)
    nn = Dense(16, activation='relu')(nn)
    nn = Dropout(0.1)(nn)
    nn = BatchNormalization()(nn)
    nn = Dense(4, activation='relu')(nn)
    nn = BatchNormalization()(nn)
    return Model(input_layer, nn)

# Early stopping 
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss',      
    patience=20,              
    restore_best_weights=True
)

## Train the Siamese network
def Batch_siamese_FNN(train_data):
    
    # Collect all races in a list
    races = train_data['RaceID'].unique()

    # Initialize lists to store features and labels for all races
    all_race_x = []
    all_race_y = []

    # Loop through unique 'RaceID' values
    for race_id in races:
        # Filter the data for the current race
        race_data = train_data[train_data['RaceID'] == race_id]
        
        # Split into response and explanatory sets
        race_y = race_data['BeatenMargin']
        race_x = race_data.drop('BeatenMargin', axis=1)
        
        # Append the features and labels to the lists
        all_race_x.append(race_x)
        all_race_y.append(race_y)

    # Concatenate features and labels for all races
    all_race_x = np.concatenate(all_race_x, axis=0)
    all_race_y = np.concatenate(all_race_y, axis=0)

    num_horses_in_race = all_race_x.shape[1]
    input_shape = all_race_x.shape[1]

    # Create a list to hold base networks for each horse
    base_networks = [create_base_network(Input(shape=(input_shape,))) for _ in range(num_horses_in_race)]

    # Define input layers for each horse
    input_layers = [Input(shape=(input_shape,)) for _ in range(num_horses_in_race)]

    # Define embeddings for each horse
    horse_embeddings = [base(input_layer) for base, input_layer in zip(base_networks, input_layers)]

    # Calculate L1 distances between horse embeddings
    L1_distances = []
    for i in range(num_horses_in_race - 1):
        for j in range(i + 1, num_horses_in_race):
            L1_distances.append(tf.abs(horse_embeddings[i] - horse_embeddings[j]))

    # Concatenate the L1 distances
    if len(L1_distances) == 1:
        concatenated_distances = L1_distances[0]
    else:
        concatenated_distances = Concatenate()(L1_distances)

    x = Dense(64, activation='relu')(concatenated_distances)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)   
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(4, activation='relu')(x)
    x = BatchNormalization()(x)

    # Output layer for predicted values
    prediction = Dense(1, activation='linear')(x)

    # Create the Siamese model
    siamese_model = Model(inputs=input_layers, outputs=prediction)

    # Compile the model
    siamese_model.compile(optimizer='adam', loss='MSE', metrics=['MSE'])

    # Train the model for all races
    siamese_model.fit([all_race_x for _ in range(num_horses_in_race)], all_race_y, epochs=25, batch_size=256, callbacks = early_stopping_callback)

    ## Save trained model
    siamese_model.save("siamese_model", save_format="tf")

    return siamese_model


## Test model performance
def validate_siamese_FNN(validation_data):

    ## Load trained model
    trained_siamese_model = load_model('siamese_model', compile=False)
    
    # Collect all races in a list in the same manner as the training process
    races = validation_data['RaceID'].unique()

    # Initialize lists to store features and labels for all races
    all_race_x = []

    # Loop through unique 'RaceID' values
    for race_id in races:
        # Filter the data for the current race
        race_data = validation_data[validation_data['RaceID'] == race_id]
        
        # Remove response
        race_x = race_data.drop('BeatenMargin', axis=1)
        
        # Append the features and labels to the lists
        all_race_x.append(race_x)

    # Concatenate features for all races
    val_x = np.concatenate(all_race_x, axis=0)

    num_horses_in_race = val_x.shape[1]

    # Make predictions on the validation dataset
    val_predictions = trained_siamese_model.predict([val_x for _ in range(num_horses_in_race)])

    return val_predictions