import numpy as np
import tensorflow as tf
import os

Y = np.load('../mfcc_pairs_labels_sg.npy', allow_pickle=False)
X = np.load('../mfcc_pairs_sg.npy', allow_pickle=False)
print(X.shape)
print(Y.shape)
X = np.asarray(X).transpose(0, 2, 1)
Y = np.asarray(Y).transpose(0, 2, 1)
print(X.shape)
print(Y.shape)
checkpoint_path = "training_30_sg_rms/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    period=5)

embeddingDim = 30
timesteps = 154
n_features = 13

model_tf = tf.keras.models.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
         units=embeddingDim, return_sequences=False), input_shape=(timesteps, n_features)),
    #tf.keras.layers.LSTM(
    #    units=embeddingDim, return_sequences=False, input_shape=(timesteps, n_features)),
    tf.keras.layers.RepeatVector(timesteps),
    tf.keras.layers.LSTM(embeddingDim, activation='tanh',
                         return_sequences=True),
    # SeqSelfAttention(embeddingDim,attention_activation='sigmoid',return_attention=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
])


model_tf.compile(optimizer='rmsprop',
                 loss=tf.losses.mean_squared_error,
                 metrics=['accuracy'])

model_tf.save_weights(checkpoint_path.format(epoch=0))
model_tf.fit(X, Y, epochs=20, batch_size=1000, callbacks=[cp_callback])
model_tf.save('Model30_sg_rms')
