# shape
# input: (None, MAX_TOKEN_LEN)

# hyperparams
VOCAB_SIZE = 300
MAX_TOKEN_LEN = 20
EMBED_DIM = 20 # fastText_model.wv.vector_size
ENCODER_HIDDEN_DIM = DECODER_HIDDEN_DIM = 512
LATENT_DIM = 256
N_CLUSTER = 10
CLUSTER_LOSS_LAMBDA = 0.5

# sampling method
def sampling(args):
    z_mea, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[[1]
    # default, random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
# VAE = encoder + decoder
# encoder model
encoder_input = layers.Input((MAX_TOKEN_LEN, ))
embedding_layer = layers.Embedding(VOCAB_SIZE, EMBED_DIM, trainable=True, weights=[initial_embedding_matrix])
encoder_input_embedding = embedding_layer(encoder_input)
encoder_lstm_layer, encoder_state_h, encoder_state_c = layers.LSTM(ENCODER_HIDDEN_DIM, return_state=True)(encoder_input_embedding)

z_mean = layers.Dense(LATENT_DIM, name='z_mean')(encoder_lstm_layer)
z_log_var = layers.Dense(LATENT_DIM, name='z_mean')(encoder_lstm_layer)
z_sampling = layers.core.Lambda(sampling, output_shape=(LATENT_DIM, ), name='z')([z_mean, z_log_var])

#instantiate encoder model
encoder = models.Model(encoder_input, [z_mean, z_log_var, z_sampling, encoder_state_h, encoder_state_c], name='encdoer')
encoder.summary()

# build decoder model
decoder_input = layers.Input((LATENT_DIM, ), name='z_smapling')
latent_repeat_layer  layers.RepeatVector(MAX_TOKEN_LEN)(decoder_input)
state_h_input = layers.Input((ENCODER_HIDDEN_DIM, ), name'encoder_state_h')
state_c_input = layers.Input((ENCODER_HIDDEN_DIM, ), name'encoder_state_c')

decoder_hidden_layer, _, _ = layers.LSTM(DECODER_HIDDEN_DIM, retrun_sequences=True, return_state=True)(latent_repeat_layer)
decoder_lstm_layer, _, _ = layers.LSTM(EMBED_DIM, return_sequences=True, return_state=True)(decoder_hidden_layer)
decooder_output = layers.wrappers.TimeDistributed(layers.Dense(EMBED_DIM))(decoder_lstm_layer)

decoder = models.Model([decoder_input, state_h_input, state_c_input], decoder_output, name='decoder')
decoder.summary()

# instantiate VAE model
vae_output = decoder(encoder(encoder_input)[2:])
vae = models.Model(encoder_input, vae_output, name='vae')
vae.summary()

# loss = reconstruction_loss + cluster_ss_loss
reconstruction_loss = K.sum(objectives.mse(encoder_input_embedding, final_output), axis=-1)

cluster_node = K.variable(np.random.normal(loc=0.0, scale=1.0, size=(N_CLUSTER, LATENT_DIM)), dtype='float32')
z_mean_repeat = K.repeat(z_mean, N_CLUSTER)
_cluster_node_repeat = K.repeat(cluster_node, K.shape(z_mean)[0])
cluster_node_repeat = K.permute_dimensions(_cluster_node_repeat, (1,0,2))
cluster_sse = K.sum(K.square(z_mean_repeat - cluster_node_repeat), axis=-1)
closest_cluster_idx = K.argmin(cluster_sse)
closest_cluster = K.map_fn((lambda idx: cluster_node[idx], closest_cluster_idx, dtype="float32")
cluster_ss_loss = K.sum(K.square(z_mean - closest_cluster), axis=-1)
vae_loss = K.mean(reconstruction_loss) - CLUSTER_LOSS_LAMBDA * K.sum(cluster_ss_loss)
