# Building a ChatBot with Deep NLP

import numpy as np

from build_model import model_inputs
from preprocessing import questionswords2int, answersints2word
from Params import maximum_length, batch_size
from train_model import test_predictions
from test_model import session, convert_string2int

# Loading the model inputs
inputs, _, _, keep_prob = model_inputs()

# Setting up the chat
while True:
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (maximum_length - len(question))
    fake_batch = np.zeros((batch_size, maximum_length))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = 'I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break

    print('ChatBot: ' + answer)
