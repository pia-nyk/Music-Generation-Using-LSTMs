from music21 import converter, instrument, chord, note
import numpy as np
from glob import glob
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.callbacks import ModelCheckpoint

notes = []

for file in glob("midi_songs/balamb.mid"):
    midi = converter.parse(file)
    notes_to_parse = None

    parts = instrument.partitionByInstrument(midi)

    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif  isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# with open('notes.txt',  'w') as f:
#     for items in notes:
#         f.write(items)

sequence_length = 100

n_vocab = len(set(notes))
print ("n_vocab = ", n_vocab)
with open('common_variables.txt', 'w') as f:
    f.write("n_vocab: %s\n" % n_vocab)
#get all pitch names
pitchnames = sorted(set(item for item in notes))

with open('pitchnames.txt', 'w') as f:
    for item in pitchnames:
        f.write("%s\n" % item)

#create  dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input =  []
network_output = []

#create input sequences and corresponding outputs
for i in range(0, len(notes)-sequence_length,1):
    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i+sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append([note_to_int[sequence_out]])

n_patterns = len(network_input)

#reshape input into a format compatible with LSTM layers
network_input_reshaped = np.reshape(network_input, (n_patterns, sequence_length, 1))

with open('common_variables.txt', 'a') as f:
    for items in network_input:
        for ints in items:
            f.write("%i," %ints)
        f.write("\n")

#normalize
network_input_reshaped = network_input_reshaped /n_vocab

network_output = np_utils.to_categorical(network_output)

#LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(network_input_reshaped.shape[1], network_input_reshaped.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]
# model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)

#load the weights
model.load_weights("weights.hdf5")
