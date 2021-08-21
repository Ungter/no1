import numpy as np 
from numpy import array
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from pickle import load

filename = 'FlickrText/Flickr_8k.trainImages.txt'
images = 'Flickr8k_Dataset/Flickr8k_Dataset'
def load_doc(filename):
    with open(filename) as f:
        for line in f:
            text = f.read()
            return text
    f.close()

def load_set(filename):
    doc = load_doc(filename)
    dset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identi = line.split('.')[0]
        dset.append(identi)
    return set(dset)

def load_clean_desc(filename, dset):
    doc = load_doc(filename)
    desc = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        img_id, img_desc = tokens[0], tokens[1:]
        if img_id in dset:
            if img_id not in desc:
                desc[img_id] = list()
            des = 'startseq' + ' '.join(img_desc) + 'endseq'
            desc[img_id].append(des)
    return desc

def load_photo_feats(filename, dset):
    all_feats = load(open(filename, 'rb'))
    feats = {k: all_feats[k] for k in dset}
    return feats

#train = load_set(filename)

#train_desc = load_clean_desc('descriptions_ted.txt',train)
#print('Descriptions: train=%d' % len(train_desc))

#train_feats = load_photo_feats = load_photo_feats('features.pkl', train)
#print('Photos: train=%d' % len(train_feats))


def to_lines(desc):
    all_desc = list()
    for key in desc.keys():
        [all_desc.append(d) for d in desc[key]]
    return all_desc

def create_token(desc):
    lines = to_lines(desc)
    token = Tokenizer()
    token.fit_on_texts(lines)
    return token

#token = create_token(train_desc)
#vocab_size = len(token.word_index) + 1
#print('vocab size: %d' % vocab_size)



def create_seq(token, max_length, desc, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    for key, desc_list in desc.items():
        for des in desc_list:
            seq = token.texts_to_sequences([des])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
                out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
                
    return array(X1), array(X2), array(y)



def max_length(desc):
    lines = to_lines(desc)
    return max(len(d.split()) for d in lines)

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096, ))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    #seq the model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    #decode
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation = 'relu')(decoder1)
    outputs = Dense(vocab_size, activation = 'softmax')(decoder2)
    # input_tensor = tf.convert_to_tensor([inputs1, inputs2])
    #input_tensor = Input(shape=(4096, 256, 256, ))
    #add together
    #input_tensor = Input(shape=(224, 224, 3))
    #model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # model = InceptionV3(inputs1=inputs1, weights='imagenet', include_top=True)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam')
    
    #summarize
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes = True)
    return model

# load training dataset (6K)
train = load_set(filename)
print('Dataset: %d' % len(train))
# descriptions
train_descriptions = load_clean_desc('descriptions_ted.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = load_photo_feats('features.pkl', train)
print('Photos: train=%d' % len(train_features))
# prepare tokenizer
token = create_token(train_descriptions)
vocab_size = len(token.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max_length(train_descriptions)
print('Description Length: %d' % max_length)
# prepare sequences
X1train, X2train, ytrain = create_seq(token, max_length, train_descriptions, train_features, vocab_size)

 
# dev dataset
 
# load test set
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_desc('descriptions_ted.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_feats('features.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = create_seq(token, max_length, test_descriptions, test_features, vocab_size)
 

#fitt the model

#define checkpoint callback
model = define_model(vocab_size, max_length)

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

#fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))











