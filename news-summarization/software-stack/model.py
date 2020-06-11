import tensorflow as tf 
import tensorflow_hub as hub 
import numpy as np
import os
import re
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Bidirectional, Input, Embedding 
from tensorflow.keras.layers import Dropout, Conv1D, Flatten
from tensorflow.keras.layers import Concatenate, Dot, Activation
import collections
import nltk.stem

class Model():
	## FIRST-ORDER FUNCTIONS
	def __init__(self, modelpath="../data/newspred.h5", usepath="./universal-sentence-encoder_4"):
		self.embed = hub.load(usepath)
		self.vocab = np.array([char for char in " abcdefghijklmnopqrstuvwxyz"])
		self.model = self.model_maker()
		self.model.load_weights(modelpath)

	def generate(self, text, numwords, k=3):
		text = self.text_cleaner(text)
		if (len(text)<100): return None, None, False
		state = self.embed([text]).numpy()
		start = np.zeros((1,100))
		start[0] = [np.where(self.vocab==r)[0][0] for r in text[:100]]
		stim, seq = text[:100], ""
		wordsgen = 0
		while (wordsgen<numwords):
			maxval, beamseq = self.beamer(start.copy(), state.copy(), k)
			seq+="".join([self.vocab[np.int(i)] for i in beamseq])
			start[0,:-k] = start[0,k:]
			start[0,-k:] = beamseq
			wordsgen+=np.sum(np.array(beamseq)==0)
		#Incase we overshoot numwords with numerous words in a single beam
		seq = " ".join(seq.split()[:-(wordsgen-numwords)]) if wordsgen>numwords else seq
		return stim, seq, True

	## SECOND-ORDER FUNCTONS
	def model_maker(self, latentdim=512):
		tf.keras.backend.clear_session()
		state = Input(shape=(latentdim,))
		decinput = Input(shape=(100,))
		embed_layer = Embedding(self.vocab.shape[0], self.vocab.shape[0], weights=[np.eye(self.vocab.shape[0])], 
		                           trainable=False, input_length=100)
		embedval = embed_layer(decinput)
		lstm_layer1 = LSTM(latentdim, return_sequences=True, return_state=True)
		lstm1val, _, _ = lstm_layer1(embedval, initial_state=[state, state])
		lstm1val = Dropout(0.2)(lstm1val)
		lstm_layer2 = Bidirectional(LSTM(latentdim, return_sequences=True, return_state=True))
		lstm2val, _, _, _, _ = lstm_layer2(lstm1val, initial_state=[state, state, state, state])
		lstm2val = Dropout(0.2)(lstm2val)
		lstm_layer3 = LSTM(latentdim, return_sequences=False, return_state=True)
		lstm3val, _, _ = lstm_layer3(lstm2val, initial_state=[state, state])
		lstm3val = Dropout(0.2)(lstm3val)
		dense_layer = Dense(self.vocab.shape[0], activation="softmax")
		output = dense_layer(lstm3val)
		mdl = tf.keras.models.Model(inputs=[decinput, state], outputs=output)
		mdl.compile(optimizer="adam", loss="categorical_crossentropy")
		return mdl

	def text_cleaner(self, s):
		s = re.sub("\n"," ", re.sub("[,<>@#\'\")(]","", s))
		s = re.sub("[.?%$0-9!&*+-/:;<=\[\]£]"," ", s)
		s = re.sub("[^ a-zA-Z]","",s)
		s = " ".join(np.vectorize(lambda s: s if len(s)<=3 else nltk.stem.WordNetLemmatizer().lemmatize(s))
                 (np.array(s.split())))
		return s.lower()

	def beamer(self, start, state, k, toplimit=10):
	    returnvals = collections.deque()
	    pred = self.model.predict([start, state])
	    if k==1:
	        returnvals.append(np.argmax(pred[0]))
	        return np.max(pred[0]), returnvals
	    else:
	        maxval, beamseq = None, None
	        topchoices = np.argsort(pred[0])[-toplimit:]
	        for j in topchoices:
	            chars = start.copy()
	            chars[0,:-1] = chars[0,1:]
	            chars[0,-1] = j
	            val, shortseq = self.beamer(chars, state, k-1)
	            if (not maxval) or ((val*pred[0,j])>maxval):
	                maxval = val*pred[0,j]
	                beamseq = shortseq
	                beamseq.appendleft(j)
	        return maxval, beamseq


