import numpy as np
import tensorflow as tf

from datetime import time

import Params
from build_model import model_inputs, seq2seq_model
from Params import maximum_length
from preprocessing import answerswords2int, questionswords2int, sorted_clean_questions, sorted_clean_answers
from Params import *

################ PART 3 - TRAINING THE SEQ2SEQ MODEL ################


# Setting the Hyperparameters
epochs = Params.epochs
batch_size = Params.batch_size
rnn_size = Params.rnn_size
num_layers = Params.num_layers
encoding_embedding_size = Params.encoding_embedding_size
decoding_embedding_size = Params.decoding_embedding_size
learning_rate = Params.learning_rate
learning_rate_decay = Params.learning_rate_decay
min_learning_rate = Params.min_learning_rate
keep_probability = Params.keep_probability

# Defining a session
tf.compat.v1.reset_default_graph()
session = tf.compat.v1.InteractiveSession()

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length
sequence_length = tf.compat.v1.placeholder_with_default(maximum_length, None, name="sequence_length")

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


# Padding the sequences with the <PAD> token
# Question: [ 'Who', 'are', 'you']
# Answer: [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
# -> Question: [ 'Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>]
# -> Answer: [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD>]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index: start_index + batch_size]
        answers_in_batch = answers[start_index: start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatboy_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_anwsers_in_batch) in enumerate(
            split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],
                                                   {inputs: padded_questions_in_batch,
                                                    targets: padded_anwsers_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_anwsers_in_batch.shape[1],
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print(
                "Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds".format(
                    epoch,
                    epochs,
                    batch_index,
                    len(training_questions) // batch_size,
                    total_training_loss_error / batch_index_check_training_loss,
                    int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_anwsers_in_batch) in enumerate(
                    split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error,
                                                          {inputs: padded_questions_in_batch,
                                                           targets: padded_anwsers_in_batch,
                                                           lr: learning_rate,
                                                           sequence_length: padded_anwsers_in_batch.shape[1],
                                                           keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
                ending_time = time.time()
                batch_time = ending_time - starting_time
                average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
                print("Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds".format(
                    average_validation_loss_error,
                    int(batch_time)))
                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate
                list_validation_loss_error.append(average_validation_loss_error)
                if average_validation_loss_error <= min(list_validation_loss_error):
                    print("I speak better now!!")
                    early_stopping_check = 0
                    saver = tf.train.Saver()
                    saver.save(session, checkpoint)
                else:
                    print("Sorry I do not speak better, I need to practice more.")
                    early_stopping_check += 1
                    if early_stopping_check == early_stopping_stop:
                        break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
