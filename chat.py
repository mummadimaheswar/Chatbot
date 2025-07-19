import numpy as np
import pandas as pd
import tensorflow as tf
import re
import time
import os

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, concatenate, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
# Load the data
lines = open('/content/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('/content/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
#     print(_line)
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
#     print(_line)
    convs.append(_line.split(','))
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

# Compare lengths of questions and answers
print(len(questions))
print(len(answers))
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = " ".join(text.split())
    return text
# Clean the data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
print("======Original questions and answers=======")
print(questions[2])
print(answers[2])

print("\n======Cleaned questions and answers===")
print(clean_questions[2])
print(clean_answers[2])
# Find the length of sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))
# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])

print(lengths['counts'].describe())

print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))
VOCAB_SIZE = 20000 #50000#15000 #14999 #to decide the dimension of sentence’s one-hot vector
EMBEDDING_DIM = 100 #to decide the dimension of Word2Vec
MAX_LEN = 20  #to unify the length of the input sentences
NUM_SAMPLES = 60000 #60000  # Number of samples to train on.
GLOVE_DIR = '/kaggle/input/glove-global-vectors-for-word-representation'
EPOCHS=40 #10
BATCH_SIZE=256
# Remove questions and answers that are shorter than 1 word and longer than 20 words.
min_line_length = 2
max_line_length = 20

# Filter out the questions that are too short/long
short_questions_temp = []
short_answers_temp = []

for i, question in enumerate(clean_questions):
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])

# Filter out the answers that are too short/long
short_questions = []
short_answers = []

for i, answer in enumerate(short_answers_temp):
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])

print(len(short_questions))
print(len(short_answers))


r = np.random.randint(1,len(short_questions))
for i in range(r, r+3):
    print(short_questions[i])
    print(short_answers[i])
    print()
#choosing number of samples
# encoder_input_text = clean_questions[:NUM_SAMPLES]
encoder_input_text = short_questions[:NUM_SAMPLES]
def tagger(input_text):
  bos = "<BOS> "
  eos = " <EOS>"
  final_target = [bos + text + eos for text in input_text]
  return final_target

# clean_answers = clean_answers[:NUM_SAMPLES]
# decoder_input_text = tagger(clean_answers)
# decoder_input_text[:5]
short_answers = short_answers[:NUM_SAMPLES]
decoder_input_text = tagger(short_answers)
decoder_input_text[:5]
print(len(encoder_input_text))
print(len(decoder_input_text))
tokenizer = Tokenizer(num_words= VOCAB_SIZE, filters='')

def vocab_creater(text_lists, VOCAB_SIZE):
  tokenizer.fit_on_texts(text_lists)
  dictionary = tokenizer.word_index

  word2idx = {}
  idx2word = {}
  for k, v in dictionary.items():
      if v < VOCAB_SIZE:
          word2idx[k] = v
          idx2word[v] = k
      if v >= VOCAB_SIZE-1:
          continue

  return word2idx, idx2word

word2idx, idx2word = vocab_creater(text_lists=encoder_input_text+decoder_input_text, VOCAB_SIZE=VOCAB_SIZE)

#print first few key/value pairs
word2idx_first5pairs = {k: word2idx[k] for k in list(word2idx)[:5]}
print('word2idx: ', word2idx_first5pairs)

idx2word_first5pairs = {k: idx2word[k] for k in list(idx2word)[:5]}
print('\nidx2word: ', idx2word_first5pairs)

# Check the length of the dictionaries.
print('word2idx length', len(word2idx))
print('idx2word length', len(idx2word))
def text2seq(tokenizer, encoder_text, decoder_text, VOCAB_SIZE):

#   tokenizer = Tokenizer(num_words=VOCAB_SIZE)
  encoder_sequences = tokenizer.texts_to_sequences(encoder_text)
  decoder_sequences = tokenizer.texts_to_sequences(decoder_text)

  return encoder_sequences, decoder_sequences

encoder_sequences, decoder_sequences = text2seq(tokenizer, encoder_input_text, decoder_input_text, VOCAB_SIZE)
print('encoder_sequences:\n', encoder_sequences[:5])
print('\ndecoder_sequences:\n', decoder_sequences[:5])
def padding(encoder_sequences, decoder_sequences, MAX_LEN):

  encoder_input_data = pad_sequences(encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
  decoder_input_data = pad_sequences(decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

  return encoder_input_data, decoder_input_data

encoder_input_data, decoder_input_data = padding(encoder_sequences, decoder_sequences, MAX_LEN)
print('encoder_input_data:\n', encoder_input_data)
print('\ndecoder_input_data:\n', decoder_input_data)
num_samples = len(encoder_sequences)

def decoder_output_creator(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE):
  decoder_output_data = np.zeros((num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
  for i, seqs in enumerate(decoder_input_data):
      for t, seq in enumerate(seqs):
          if t > 0:
                decoder_output_data[i][t][seq] = 1.  #decoder_output_data[i, t, seq] = 1.

  return decoder_output_data

decoder_output_data = decoder_output_creator(decoder_input_data, num_samples, MAX_LEN, VOCAB_SIZE)
print (encoder_input_data.shape)
print (decoder_input_data.shape)
print (decoder_output_data.shape)
# Download the glove model
!wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
!unzip -q glove.twitter.27B.zip

# Update GLOVE_DIR to the correct path
GLOVE_DIR = '/content/'
print(f"GLOVE_DIR is now set to: {GLOVE_DIR}")
# Call Glove file
def glove_100d_dictionary(glove_dir):
  embeddings_index = {}
  f = open(os.path.join(glove_dir, 'glove.twitter.27B.100d.txt'))
  for line in f:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs
  f.close()
  return embeddings_index

embeddings_index = glove_100d_dictionary(GLOVE_DIR)
print('Found %s word vectors.' % len(embeddings_index))
#Create Embedding Matrix from our Vocabulary
def embedding_matrix_creater(max_words, embedding_dimension):
  embedding_matrix = np.zeros((max_words, embedding_dimension))  #np.zeros((len(word2idx) + 1, embedding_dimension))
  for word, i in word2idx.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
  return embedding_matrix

embedding_matrix = embedding_matrix_creater(VOCAB_SIZE, EMBEDDING_DIM)
print(embedding_matrix.shape)
print((len(word2idx) + 1))
#Create Embedding Layer
def embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix):

  embedding_layer = Embedding(input_dim = VOCAB_SIZE,
                              output_dim = EMBEDDING_DIM,
                              input_length = MAX_LEN,
                              weights = [embedding_matrix],
                              trainable = False)
  return embedding_layer

embedding_layer = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, MAX_LEN, embedding_matrix)

embedding_layer2 = embedding_layer_creater(VOCAB_SIZE, EMBEDDING_DIM, None, embedding_matrix)
def data_spliter(encoder_input_data, decoder_input_data, test_size1=0.2, test_size2=0.3):
  en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data, test_size=test_size1)
  en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)

  return en_train, en_val, en_test, de_train, de_val, de_test

en_train, en_val, en_test, de_train, de_val, de_test = data_spliter(encoder_input_data, decoder_input_data)
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, concatenate, TimeDistributed

def build_seq2seq_model(HIDDEN_DIM=300):
    #set up the encoder
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embedding_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(decoder_dense)(decoder_outputs)

    #create seq2seq model
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc']) #sparse_categorical_crossentropy as labels in a single integer array

    #create encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    #Create sampling/decoder model
    decoder_state_input_h  = Input(shape=(HIDDEN_DIM,))
    decoder_state_input_c = Input(shape=(HIDDEN_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding2= embedding_layer(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_LSTM(decoder_embedding2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

def build_seq2seq_model2(HIDDEN_DIM=300):
    #set up the encoder
    encoder_inputs = Input(shape=(None, ), dtype='int32',)
    encoder_embedding = embedding_layer2(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, ), dtype='int32',)
    decoder_embedding = embedding_layer2(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(decoder_dense)(decoder_outputs)

    #create seq2seq model
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc']) #sparse_categorical_crossentropy as labels in a single integer array

    #create encoder model
    encoder_model = Model(encoder_inputs, encoder_states)

    #Create sampling/decoder model
    decoder_state_input_h  = Input(shape=(HIDDEN_DIM,))
    decoder_state_input_c = Input(shape=(HIDDEN_DIM,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_embedding2= embedding_layer2(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_LSTM(decoder_embedding2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

    return model, encoder_model, decoder_model

model, encoder_model, decoder_model = build_seq2seq_model2(HIDDEN_DIM=300)
model.summary()

# model, encoder_model, decoder_model = build_seq2seq_model(HIDDEN_DIM=300)
# model.summary()
encoder_model.summary()
decoder_model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#early stopping & saving
# es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
mcp = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05,
          callbacks= [mcp]
         )
def translate_sentence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = word2idx['<bos>']
    eos = word2idx['<eos>']
    output_sentence = []

    for _ in range(MAX_LEN):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]  # Update states

    return ' '.join(output_sentence)

for index in range(10):
    i = np.random.randint(1, len(encoder_input_data))
    input_seq = encoder_input_data[i:i+1]
    translation = translate_sentence(input_seq)
    print('-')
    print('Input:', short_answers[i])
    print('Response:', translation)
