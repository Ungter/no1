from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

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

def max_length(desc):
    lines = to_lines(desc)
    return max(len(d.split()) for d in lines)

def word_for_id(integer, token):
    for word, index in token.word_index.items():
        if index == integer:
            return word
        return None

def gene_desc(model, token, photo, max_len):
    # seed the genertation process
    in_text = 'startseq'
    # iterate over the whole len of the seq
    for i in range(max_len):
        # int encode input seq
        seq = token.texts_to_sequences([in_text])[0]
        # pad input
        seq = pad_sequences([seq], maxlen = max_len)
        # predict the next word
        yhat = model.predict([photo, seq], verbose = 0)
        # convert probability to a int
        yhat = argmax(yhat)
        # map the int to word
        word = word_for_id(yhat, token)
        # stop if the word cannot be mapped
        if word is None:
            break
        #append as each word is being generated
        in_text += ' ' + word
        # stop if predicts end of seq
        if word == 'endseq':
            break
    return in_text

def eva_model(model, desc, photos, token, max_len):
    actual, predicted = list(), list()
    # step over the entire set
    for key, desc_list in desc.items():
        #gene desc
        yhat = gene_desc(model, token, photos[key], max_len)
        #store actual and predicted
        ref = [d.split() for d in desc_list]
        actual.append(ref)
        predicted.append(yhat.split())
        cc = SmoothingFunction()
    print('bleu1: %f' % corpus_bleu(actual, predicted, weights = (1.0, 0, 0, 0), smoothing_function=cc.method4))
    print('bleu2: %f' % corpus_bleu(actual, predicted, weights = (0.5, 0.5, 0, 0), smoothing_function=cc.method4))
    print('bleu3: %f' % corpus_bleu(actual, predicted, weights = (0.33, 0.33, 0.33, 0), smoothing_function=cc.method4))
    print('bleu4: %f' % corpus_bleu(actual, predicted, weights = (0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method4))

# load training dataset (6K)
filename = 'FlickrText/Flickr_8k.trainImages.txt'
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

#prep test set

filename = 'FlickrText/Flickr_8k.testImages.txt'
test = load_set(filename)
print('dataset: %d' % len(test))
#desc
test_desc = load_clean_desc('descriptions_ted.txt', test)
print('desc: test=%d' % len(test_desc))
#photo feats
test_feats = load_photo_feats('features.pkl', test)
print('photo: test=%d' % len(test_feats))

#load in the model
filename = 'model-ep020-loss3.431-val_loss3.118.h5'
model = load_model(filename)
#eva model
eva_model(model, test_desc, test_feats, token, max_length)