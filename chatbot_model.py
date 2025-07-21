# chatbot_model.py
import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
MAX_LEN = 20
HIDDEN_DIM = 300

def clean_text(text):
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
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return " ".join(text.split())

def tagger(texts):
    return [f"<bos> {t} <eos>" for t in texts]

def build_tokenizer(texts, vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(num_words=vocab_size, filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def build_seq2seq(vocab_size, embedding_dim, hidden_dim):
    encoder_inputs = Input(shape=(MAX_LEN,))
    decoder_inputs = Input(shape=(MAX_LEN,))

    embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
    encoder_emb = embedding(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(hidden_dim, return_state=True)(encoder_emb)
    encoder_states = [state_h, state_c]

    decoder_emb = embedding(decoder_inputs)
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_emb, initial_state=encoder_states)

    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(hidden_dim,))
    decoder_state_input_c = Input(shape=(hidden_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    dec_emb2 = embedding(decoder_inputs)
    dec_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=decoder_states_inputs)
    dec_outputs2 = decoder_dense(dec_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [dec_outputs2, state_h2, state_c2])

    return model, encoder_model, decoder_model

def translate_sentence(input_seq, encoder_model, decoder_model, word2idx, idx2word, max_len=MAX_LEN):
    states_value = encoder_model.predict(input_seq)
    bos = word2idx.get('<bos>', 1)
    eos = word2idx.get('<eos>', 2)

    target_seq = np.array([[bos]])
    output_sentence = []

    for _ in range(max_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == eos or sampled_token_index == 0:
            break
        output_sentence.append(idx2word.get(sampled_token_index, '<unk>'))
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]
    return ' '.join(output_sentence)
