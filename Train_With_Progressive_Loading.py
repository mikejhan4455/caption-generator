import math
import glob
import os
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler, EarlyStopping
from tensorflow.test import is_gpu_available
from sru import SRU


# Tensor board plot learning rate
class LRTensorBoard(TensorBoard):

    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return lrate


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# load a pre-defined list of photo identifiers


def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

# load clean descriptions into memory


def load_clean_descriptions(filename, dataset):
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            descriptions[image_id].append(desc)

    return descriptions

# load photo features


def load_photo_features(filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

# covert a dictionary of clean descriptions to a list of descriptions


def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# fit a tokenizer given caption descriptions


def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# calculate the length of the description with the most words


def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

# create sequences of images, input sequences and output words for an image


def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)

# define the captioning model


def define_model(vocab_size, max_length, load_checkpoint=False, checkpoint_path=None):
    # feature extractor model
    if not load_checkpoint:
        print('generate new model')

        inputs1 = Input(shape=(4096,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(512, activation='relu')(fe1)
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 512, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = SRU(512)(se2)

        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(512, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)

        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    # resume from checkpoint
    else:
        print('load model from checkpoint: {}'.format(checkpoint_path))
        model = load_model(checkpoint_path)

    # summarize model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    # compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model

# data generator, intended to be used in a call to model.fit_generator()


def data_generator(descriptions, photos, tokenizer, max_length, data_amount):
    # loop for ever over images
    while 1:
        for key, desc_list in list(descriptions.items())[data_amount:]:
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]


def val_data_generator(descriptions, photos, tokenizer, max_length, data_amount):
    # loop for ever over images
    while 1:
        for key, desc_list in list(descriptions.items())[:data_amount]:
            # retrieve the photo feature
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(
                tokenizer, max_length, desc_list, photo)
            yield [[in_img, in_seq], out_word]


# start here
is_gpu_available()


# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
assert 'GPU' in str(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
assert len(backend.tensorflow_backend._get_available_gpus()) > 0


# load training dataset (6K)
filename = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))

# descriptions
train_descriptions = load_clean_descriptions('flk_des.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))

# photo features
train_features = load_photo_features('flk_features.pkl', train)
print('Photos: train=%d' % len(train_features))

# prepare tokenizer
tokenizer = create_tokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)

# define the model
model = define_model(vocab_size, max_length, load_checkpoint=True,
                     checkpoint_path='./checkpoints/bak/2-34.hdf5')
# train the model, run epochs manually and save after each epoch

RUNS = 10
EPOCHS = 99
samples = len(train_descriptions)
val_split = 0.2
val_d_amount = int(samples * val_split)
BATCH_SIZE = samples


checkpoint_path = "./checkpoints/r{runs:02d}/e{{epoch:02d}}-{{loss:.4f}}.hdf5"

# create the validatoin data generator @ pre 20% of training data
val_data = val_data_generator(
    train_descriptions, train_features, tokenizer, max_length, val_d_amount)

# create the training data generator @ post 80% of training data
training_data = data_generator(
    train_descriptions, train_features, tokenizer, max_length, val_d_amount)


# fit in runs
for r in range(RUNS):

    try:
        os.mkdir('./checkpoints/r{:02d}'.format(r))
    except Exception:
        pass

    checkpoint = ModelCheckpoint(
        checkpoint_path.format(runs=r), monitor='val_loss', verbose=0, save_best_only=False, mode='min')

    early_stoping = EarlyStopping(
        monitor='val_loss', mode='auto', patience=3)

    # check wether the folder exist
    log_dir = './logs/{}/'.format(r)
    if log_dir not in os.listdir('./logs/'):
        os.mkdir(log_dir)

    callbacks_list = [checkpoint, LRTensorBoard(log_dir=log_dir),
                      LearningRateScheduler(step_decay), early_stoping]

    model.fit_generator(training_data, epochs=EPOCHS, samples_per_epoch=BATCH_SIZE,
                        validation_data=val_data, validation_steps=1, verbose=1, callbacks=callbacks_list)

    # early stopping, load latest weigths and start a new run
    latest_chekpoint = max(
        glob.glob('./checkpoints/r{:02d}/e??-??????.hdf5'.format(r)))
    model = define_model(vocab_size, max_length,
                         load_checkpoint=True, checkpoint_path=latest_chekpoint)
