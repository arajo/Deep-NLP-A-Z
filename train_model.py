import tensorflow as tf

from build_model import model_inputs, seq2seq_model
from Params import MAXIMUM_LENGTH
from preprocessing import answerswords2int, questionswords2int

################ PART 3 - TRAINING THE SEQ2SEQ MODEL ################


# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.placeholder_with_default(MAXIMUM_LENGTH, None, name="sequence_length")

# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int
                                                       )

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in
                         gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
