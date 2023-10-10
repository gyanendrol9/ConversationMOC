# Modelling
from data_reader import *
from plots import *

import numpy as np
from keras import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Concatenate, Bidirectional, Flatten

from keras.callbacks import EarlyStopping
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from stellargraph.layer import GraphConvolution

from keras.layers import Attention, MultiHeadAttention
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import tensorflow as tf

import os
import gc

epochs = 200
batch_size = 64

# Masked loss function
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        mask = K.sum(y_true[..., :3], axis=-1) #sum the target_user label vector
        bool_mask = np.sum(y_true[..., 0:3], axis=-1) != 0 
        y_true_masked = y_true[bool_mask] # masked the non-target_user true_label
        y_pred_masked = y_pred[bool_mask] # masked the non-target_user pred_label
        # Convert the true labels to one-hot encoding
#         y_true = tf.one_hot(tf.cast(y_true, dtype=tf.int32), depth=tf.shape(y_pred)[-1])
        
        # Calculate the softmax probabilities
#         y_true = K.softmax(y_true)
        y_pred_prob = K.softmax(y_pred_masked)
        
        # Calculate the focal loss
        loss = -alpha * (1 - y_pred_prob) ** gamma * y_true_masked * K.log(y_pred_prob + K.epsilon())
        loss = K.mean(K.sum(loss, axis=-1))
        
        return loss
    
    return loss

def masked_categorical_crossentropy(y_true, y_pred):
    mask = K.sum(y_true[..., :3], axis=-1) #sum the target_user label vector
    bool_mask = np.sum(y_true[..., 0:3], axis=-1) != 0 
    y_true_masked = y_true[bool_mask] # masked the non-target_user true_label
    y_pred_masked = y_pred[bool_mask] # masked the non-target_user pred_label
    
    loss = K.categorical_crossentropy(y_true_masked, y_pred_masked)
    return K.sum(loss) / K.sum(mask)


def focal_loss_crossentropy(y_true, y_pred, gamma=2.0, alpha=[0.2,0.25,0.5,0.05]):
    alpha = np.asarray(alpha)
    mask = K.sum(y_true[..., :3], axis=-1) #sum the target_user label vector
    bool_mask = np.sum(y_true[..., 0:3], axis=-1) != 0 
    y_true_masked = y_true[bool_mask] # masked the non-target_user true_label
    y_pred_masked = y_pred[bool_mask] # masked the non-target_user pred_label

    # Calculate the focal loss
    loss = -alpha * (1 - y_pred_masked) ** gamma * y_true_masked * K.log(y_pred_masked + K.epsilon())
    loss = K.mean(K.sum(loss, axis=-1))

    return loss


def get_lstm_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes):

    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # Define LSTM layer
    # lstm_output = Bidirectional(LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1))(input_node_features)
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)

    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="dense_1")(lstm_output)  # softmax output layer task1

    # Create model
    model = Model(inputs=input_node_features, outputs=out)
    return model

def get_lstm_model_task2(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes):

    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # Define LSTM layer
    # lstm_output = Bidirectional(LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1))(input_node_features)
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)
    
    layer = MultiHeadAttention(num_heads=8, key_dim=2)
    attn_input_node_features, _ = layer(lstm_output, lstm_output, return_attention_scores=True)
    out = Flatten()(attn_input_node_features)
    out = Dropout(0.1)(out)
    out = Dense(num_classes,activation='softmax', name="dense_2")(out)  # softmax output layer task2

    # Create model
    model = Model(inputs=input_node_features, outputs=out)
    return model

def get_lstm_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2):
    
    # Define input layers
    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # LSTM layer
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)

    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="dense_1")(lstm_output)  # softmax output layer task1
    
    layer = MultiHeadAttention(num_heads=8, key_dim=2)
    attn_input_node_features, _ = layer(lstm_output, lstm_output, return_attention_scores=True)
    out2 = Flatten()(attn_input_node_features)
    out2 = Dropout(0.1)(out2)
    out2 = Dense(num_classes2,activation='softmax', name="dense_2")(out2)  # softmax output layer task2
    
    # Create model
    model = Model(inputs=input_node_features, outputs=[out,out2])
    
    return model

def get_lstm_gcn_multiplex_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_net=2):
    
    # Define input layers
    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # Define multiplex layers input 
    input_networks = []
    for n in range(num_net):
        input_networks.append(Input(shape=(input_posts_timeline,input_posts_timeline)))

    # LSTM layer
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)

    # Define GCN layer
    gcn_networks = []
    gcn1 = GraphConvolution(units=output_dim_node_features)
    gcn2 = GraphConvolution(units=output_dim_node_features)
    gcn_attn_layer = MultiHeadAttention(num_heads=8, key_dim=2)

    for n in range(num_net):
        gcn_output = gcn1([lstm_output, input_networks[n]])
        gcn_output = gcn2([gcn_output, input_networks[n]])
        gcn_networks.append(gcn_output)
    

    if num_net > 1:
        gcn_multiplex_output = Concatenate()(gcn_networks)    
    else:
        gcn_multiplex_output = gcn_networks[0]
        
    gcn_output_SAF, _ = gcn_attn_layer(gcn_multiplex_output, gcn_multiplex_output, return_attention_scores=True)
    
    concatenated_output = Concatenate()([gcn_output_SAF, lstm_output])  
    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="dense_1")(concatenated_output)  # softmax output layer task1
    
    # Create model
    model = Model(inputs=input_networks+[input_node_features], outputs=out)
        
    return model

def get_lstm_gcn_multiplex_model_task2(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_net=2):
    
    # Define input layers
    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # Define multiplex layers input 
    input_networks = []
    for n in range(num_net):
        input_networks.append(Input(shape=(input_posts_timeline,input_posts_timeline)))

    # LSTM layer
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)

    # Define GCN layer
    gcn_networks = []
    gcn1 = GraphConvolution(units=output_dim_node_features)
    gcn2 = GraphConvolution(units=output_dim_node_features)
    gcn_attn_layer = MultiHeadAttention(num_heads=8, key_dim=2)

    for n in range(num_net):
        gcn_output = gcn1([lstm_output, input_networks[n]])
        gcn_output = gcn2([gcn_output, input_networks[n]])
        gcn_networks.append(gcn_output)
    
    if num_net > 1:
        gcn_multiplex_output = Concatenate()(gcn_networks)    
    else:
        gcn_multiplex_output = gcn_networks[0]

    gcn_output_SAF, _ = gcn_attn_layer(gcn_multiplex_output, gcn_multiplex_output, return_attention_scores=True)
    
    concatenated_output = Concatenate()([gcn_output_SAF, lstm_output])  
    
    layer = MultiHeadAttention(num_heads=8, key_dim=2)
    attn_input_node_features, _ = layer(concatenated_output, concatenated_output, return_attention_scores=True)
    out = Flatten()(attn_input_node_features)
    out = Dropout(0.1)(out)
    out = Dense(num_classes,activation='softmax', name="dense_2")(out)  # softmax output layer task2

    # Create model
    model = Model(inputs=input_networks+[input_node_features], outputs=out)
        
    return model

def get_lstm_gcn_multiplex_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2, num_net=2):
    
    # Define input layers
    input_node_features = Input(shape=(input_posts_timeline, input_features_dim), name='input_node_features')

    # Define multiplex layers input 
    input_networks = []
    for n in range(num_net):
        input_networks.append(Input(shape=(input_posts_timeline,input_posts_timeline)))

    # LSTM layer
    lstm_output = LSTM(units=output_dim_node_features, return_sequences=True, recurrent_dropout=0.1)(input_node_features)
    lstm_output = Dropout(0.1)(lstm_output)

    # Define GCN layer
    gcn_networks = []
    gcn1 = GraphConvolution(units=output_dim_node_features)
    gcn2 = GraphConvolution(units=output_dim_node_features)
    gcn_attn_layer = MultiHeadAttention(num_heads=8, key_dim=2)

    for n in range(num_net):
        gcn_output = gcn1([lstm_output, input_networks[n]])
        gcn_output = gcn2([gcn_output, input_networks[n]])
        gcn_networks.append(gcn_output)
    

    if num_net > 1:
        gcn_multiplex_output = Concatenate()(gcn_networks)    
    else:
        gcn_multiplex_output = gcn_networks[0]
        
    gcn_output_SAF, _ = gcn_attn_layer(gcn_multiplex_output, gcn_multiplex_output, return_attention_scores=True)
    
    concatenated_output = Concatenate()([gcn_output_SAF, lstm_output])  
    out = TimeDistributed(Dense(num_classes, activation="softmax"), name="dense_1")(concatenated_output)  # softmax output layer task1
    
    layer = MultiHeadAttention(num_heads=8, key_dim=2)
    attn_input_node_features, _ = layer(concatenated_output, concatenated_output, return_attention_scores=True)
    out2 = Flatten()(attn_input_node_features)
    out2 = Dropout(0.1)(out2)
    out2 = Dense(num_classes2,activation='softmax', name="dense_2")(out2)  # softmax output layer task2
    
    # Create model
    model = Model(inputs=input_networks+[input_node_features], outputs=[out,out2])
    
    return model

def train_lstm_singletask(epochs, batch_size, x_train, y_train, title, kfold, model_path):
    
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train[0].shape
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask-all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(epochs, batch_size, x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask-all_emb-{title}-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model

    if not os.path.exists(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred), metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask-all_emb-FLF-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(epochs, batch_size, x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask-all_emb-{title}-FLF-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model


def train_multiplexnet_singletask(multiplex_net_train, epochs, batch_size, x_train, y_train, title, kfold, model_path):
    
    num_net = len(multiplex_net_train)
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train[0].shape
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_gcn_multiplex_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_net)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask-all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(multiplex_net_train+[x_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask-all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask-all_emb-{title}-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model

    if not os.path.exists(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_gcn_multiplex_model(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_net)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred), metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask-all_emb-FLF-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(multiplex_net_train+[x_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask-all_emb-{title}-FLF-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model

def train_lstm_singletask_B(epochs, batch_size, x_train, y_train, title, kfold, model_path):
    
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train.shape
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_model_task2(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask_B_all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(epochs, batch_size, x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask_B_all_emb-{title}-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model


def train_multiplexnet_singletask_B(multiplex_net_train, epochs, batch_size, x_train, y_train, title, kfold, model_path):
    
    num_net = len(multiplex_net_train)
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train.shape
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_gcn_multiplex_model_task2(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_net)

        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_singletask_B_all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = model.fit(multiplex_net_train+[x_train], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/singletask_B_all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_singletask_B_all_emb-{title}-{epochs}.pdf"
        plot_training_model_singletask(history, filepath=filepath, savefig=True)
        del model


def train_lstm_multitask(epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path):
    
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train[0].shape
    num_classes2 = y2_train[0].shape[0]
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2)

        if num_classes <= 2:
            loss1 = "binary_crossentropy"
        else:
            loss1 = "categorical_crossentropy" #lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred)

        if num_classes2 <= 2:
            loss2 = "binary_crossentropy"
        else:
            loss2 = "categorical_crossentropy"

        losses = {
                "dense_1": loss1,
                "dense_2": loss2}

        optimizer = Adam(lr=0.0001)
        lossWeights = {"dense_1": 1, "dense_2": 1}  # Weightage for optimizing the error loss
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                metrics=["accuracy"])
        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_multitask-all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_dense_1_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dense_1_loss', patience=10)

        history = model.fit(x_train, [y_train, y2_train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_multitask-all_emb-{title}-{epochs}.pdf"
        plot_training_model_multitask(history, filepath=filepath, savefig=True)
        del history

        del model
        gc.collect()

    if not os.path.exists(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2)

        if num_classes <= 2:
            loss1 = "binary_crossentropy"
        else:
            loss1 = lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred)

        if num_classes2 <= 2:
            loss2 = "binary_crossentropy"
        else:
            loss2 = "categorical_crossentropy"

        losses = {
                "dense_1": loss1,
                "dense_2": loss2}

        optimizer = Adam(lr=0.0001)
        lossWeights = {"dense_1": 1, "dense_2": 1}  # Weightage for optimizing the error loss
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                metrics=["accuracy"])
        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_multitask-all_emb-FLF-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_dense_1_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dense_1_loss', patience=10)

        history = model.fit(x_train, [y_train, y2_train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_multitask-all_emb-{title}-FLF-{epochs}.pdf"
        plot_training_model_multitask(history, filepath=filepath, savefig=True)

        del history

        del model
        gc.collect()


def train_multiplexnet_multitask(multiplex_net_train, epochs, batch_size, x_train, y_train, y2_train, title, kfold, model_path):
    
    num_net = len(multiplex_net_train)
    input_posts_timeline, input_features_dim = x_train[0].shape
    _, num_classes = y_train[0].shape
    num_classes2 = y2_train[0].shape[0]
    output_dim_node_features = 100 # Dimension of output node features

    if not os.path.exists(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.h5"):

        model = get_lstm_gcn_multiplex_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2, num_net)

        if num_classes <= 2:
            loss1 = "binary_crossentropy"
        else:
            loss1 = "categorical_crossentropy" #lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred)

        if num_classes2 <= 2:
            loss2 = "binary_crossentropy"
        else:
            loss2 = "categorical_crossentropy"

        losses = {
                "dense_1": loss1,
                "dense_2": loss2}

        optimizer = Adam(lr=0.0001)
        lossWeights = {"dense_1": 1, "dense_2": 1}  # Weightage for optimizing the error loss
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                metrics=["accuracy"])
        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_multitask-all_emb-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_dense_1_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dense_1_loss', patience=10)

        history = model.fit(multiplex_net_train+[x_train], [y_train, y2_train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/multitask-all_emb-{title}-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_multitask-all_emb-{title}-{epochs}.pdf"
        plot_training_model_multitask(history, filepath=filepath, savefig=True)

        del history
    
        del model
        gc.collect()

    if not os.path.exists(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5"):
        model = get_lstm_gcn_multiplex_model_multitask(input_posts_timeline, input_features_dim, output_dim_node_features, num_classes, num_classes2, num_net)

        if num_classes <= 2:
            loss1 = "binary_crossentropy"
        else:
            loss1 = lambda y_true, y_pred: focal_loss_crossentropy(y_true, y_pred)

        if num_classes2 <= 2:
            loss2 = "binary_crossentropy"
        else:
            loss2 = "categorical_crossentropy"

        losses = {
                "dense_1": loss1,
                "dense_2": loss2}

        optimizer = Adam(lr=0.0001)
        lossWeights = {"dense_1": 1, "dense_2": 1}  # Weightage for optimizing the error loss
        model.compile(optimizer=optimizer, loss=losses, loss_weights=lossWeights,
                metrics=["accuracy"])
        # Print the model summary
        model.summary()

        # Define the checkpoint filepath
        filepath = model_path+f"/Fold-{kfold}-{title}"+"_multitask-all_emb-FLF-{epoch:02d}.hdf5"
        checkpoint_callback = ModelCheckpoint(filepath, monitor='val_dense_1_loss', verbose=1, save_weights_only=True, period=10)

        # Define early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dense_1_loss', patience=10)

        history = model.fit(multiplex_net_train+[x_train], [y_train, y2_train], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint_callback, early_stop], verbose=1)
            
        model_json = model.to_json()
        with open(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(f"{model_path}/multitask-all_emb-{title}-FLF-{epochs}_fold_{kfold}.h5")

        filepath = model_path+f"/Fold-{kfold}_multitask-all_emb-{title}-FLF-{epochs}.pdf"
        plot_training_model_multitask(history, filepath=filepath, savefig=True)

        del history
        del model
        gc.collect()