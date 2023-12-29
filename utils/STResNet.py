from tensorflow import keras

import numpy as np
import kapre

import time

class Classifier:
    def __init__(self, config, nb_classes, verbose=0):
        self.config = config
        self.nb_classes = nb_classes
        
        keras.backend.clear_session()
        
        self.model = None
        self.verbose = verbose

        self.model = self.build_model()
        
    def one_chennel_resnet (self, branch_idx):
        
        num_filters = self.config['num_filters']
        kernel_sizes = self.config['kernel_sizes']
        cnn_per_res = self.config['cnn_per_res']
        max_filters = self.config['max_filters']
        signal_size = self.config['signal_size'][branch_idx]

        branch_idx = str(branch_idx)
        # add an input layer
        my_input = keras.layers.Input(shape=((signal_size, 1)), name = "input_{}".format(branch_idx))
        for i in np.arange(self.config["num_res_blocks"]):
            if(i==0):
                block_input = my_input
                x = keras.layers.BatchNormalization(name = "batch_0_{}_{}".format(branch_idx, str(i)))(block_input)
            else:
                block_input = x
            for j in np.arange(cnn_per_res):
                x = keras.layers.Conv1D(num_filters, kernel_sizes[j], padding='same', name = "conv_{}_{}_{}".format(str(j), branch_idx, str(i)))(x)
                if(j<cnn_per_res-1):
                    x =  keras.layers.LeakyReLU(name = "act_{}_{}_{}".format(str(j), branch_idx, str(i)))(x)
            is_expand_channels = not (signal_size == num_filters)
            if is_expand_channels:
                res_conn = keras.layers.Conv1D(num_filters, 1, padding='same', name = "e_conv_{}_{}".format(branch_idx, str(i)))(block_input)
            else:
                res_conn = keras.layers.BatchNormalization(name = "s_conv_{}_{}".format(branch_idx, str(i)))(block_input)
            x = keras.layers.add([res_conn, x], name = "add_{}_{}".format(branch_idx, str(i)))
            x =  keras.layers.LeakyReLU(name = "last_act_{}_{}".format(branch_idx, str(i)))(x)
            if(i<self.config["num_res_blocks"]): #perform pooling on all blocks aside from the last
                x = keras.layers.MaxPool1D(pool_size=self.config["pool_size"],
                                           strides = self.config['pool_stride_size'],
                                           name = "max_pool_{}_{}".format(branch_idx, str(i)))(x)
            num_filters = 2*num_filters
            if max_filters<num_filters:
                num_filters = max_filters

        return my_input,x
    
    def spectro_layer_mid(self, input_x, branch_idx):
        fmax=max(3,self.config['sampling_rate'][branch_idx]//2) #at least 3 frequnecy bands
        win_length = self.config['fft_win_length'][branch_idx]
        hop_length = self.config['hop_length'][branch_idx]

        branch_idx = str(branch_idx)


        x = kapre.time_frequency.STFT(n_fft = fmax, win_length= win_length, hop_length= hop_length,
                                                                  input_data_format='channels_last')(input_x)
        x = kapre.time_frequency.Magnitude()(x)
        x = kapre.time_frequency.MagnitudeToDecibel()(x)
        x = keras.layers.Conv2D(self.config['num_filters_spectro'], (3,1), padding='same', name = 'conv_0_{}_77'.format(branch_idx))(x)
        x =  keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(self.config['num_filters_spectro'], (3,1), padding='same', name = 'conv_1_{}_77'.format(branch_idx))(x)
        x =  keras.layers.LeakyReLU()(x)
        x = keras.layers.Conv2D(self.config['num_filters_spectro'], (3,1), padding='same', name = 'conv_2_{}_77'.format(branch_idx))(x)
        x =  keras.layers.LeakyReLU()(x)
        x = keras.layers.Flatten(name = 'spectro_flatten_' + branch_idx + "_77")(x)
        x = keras.layers.Dense(self.config['dense_neurons_spectro'], activation="relu",
                               kernel_regularizer=keras.regularizers.l2(self.config["l2_lambda"]),
                               name = "sectro_dense_" + branch_idx + "_77")(x)
        
        x =  keras.layers.Dropout(self.config["drop_rate"], name = 'spectro_dropout_' + branch_idx + "_77")(x)

        return x

    def build_model(self):
        inputs = []
        channel_outputs = []

        #channel specific residual layers (time-domain)
        for branch_idx in np.arange(self.config['num_channels']):
            channel_resnet_input, channel_resnet_out= self.one_chennel_resnet(branch_idx)
            channel_resnet_out = keras.layers.Flatten(name="flatten_"+str(branch_idx))(channel_resnet_out)
            channel_outputs.append(channel_resnet_out)
            inputs.append(channel_resnet_input)

        #channel spectral layers (frequency-domain)
        spectral_outputs = []
        
        for branch_idx, x in enumerate(inputs):
            channel_out = self.spectro_layer_mid(x, branch_idx)
            channel_out = keras.layers.Flatten(name='flatten_spectro_' + str(branch_idx) + "_77")(channel_out)
            spectral_outputs.append(channel_out)

        #concateante the channel specific residual layers
        x =  keras.layers.concatenate(channel_outputs,axis=-1, name='concatenate_time_88')

        #join time-domain and frequnecy domain fully-conencted layers
        s =  keras.layers.concatenate(spectral_outputs,axis=-1, name = 'concatenate_freq_88')

        x =  keras.layers.concatenate([s,x], name = 'concatenate_time_freq_88')
        x =  keras.layers.Dropout(self.config["drop_rate"], name = 'dropout_0_88')(x)
        x =  keras.layers.Dense(self.config['dense_neurons'][0], activation="linear",
                                kernel_regularizer=keras.regularizers.l2(self.config["l2_lambda"]), 
                                name = 'dense_0_88')(x)
        x =  keras.layers.LeakyReLU()(x)
        x =  keras.layers.Dropout(self.config["drop_rate"], name = 'dropout_1_88')(x)
        x =  keras.layers.Dense(self.config['dense_neurons'][0], activation="linear",
                                kernel_regularizer=keras.regularizers.l2(self.config["l2_lambda"]),
                                name = 'dense_1_88')(x)
        x =  keras.layers.LeakyReLU()(x)
        x =  keras.layers.Dropout(self.config["drop_rate"], name = 'dropout_2_88')(x)
        output =  keras.layers.Dense(self.nb_classes,activation="softmax", name = 'softmax_88')(x)

        model =  keras.models.Model(inputs=inputs, outputs=output)
        optimizer = keras.optimizers.legacy.Adam(learning_rate=self.config['learning_rate'], decay = self.config['learning_decay'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])  

        return model

    def fit(self, x_train, y_train, x_val, y_val, batch_size, epochs, callbacks):
        print ("num_epochs:", epochs, "---- batch_size:", batch_size)
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                              validation_data = (x_val, y_val), verbose = self.verbose, 
                              callbacks = callbacks)
        return hist.history
    
    def predict(self, x_test, y_test):
        predictions_binary = self.model.predict(x_test, batch_size = 64)
        groundtruth_binary = y_test

        # convert them back to a [0, 1, 2, 3] format
        predictions = np.argmax(predictions_binary, axis=1)
        groundtruth = np.argmax(groundtruth_binary, axis=1)

        return predictions, groundtruth

    def export_model (self, path):
        self.model.save_weights(path)
        
        


