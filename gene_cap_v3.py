from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model

def ext_feats(filename):
    # load in the model
    model = InceptionV3(weights='imagenet', include_top=True)
    # re-structure the model
    model = Model(inputs = model.inputs, outputs = model.layers[-2].output)
    # laod the photo
    img = load_img(filename, target_size = (224, 224))
    # convert the image pixels to numpy array
    img = img_to_array(img)
    # reshape data
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # prep the img for  vgg model
    img = preprocess_input(img)
    # get feats
    feats = model.predict(img, verbose = 0)
    return feats

def word_for_id(integer, token):
    for word, index, in token.word_index.items():
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

#load the tokenizer
token = load(open('token.pkl', 'rb'))
max_len = 34
# load the model
model = load_model('model-ep020-loss3.410-val_loss3.113.h5')
# load & prepare the photo to be identified
photo = ext_feats('beach.jpg')
# gene desc
desc = gene_desc(model, token, photo, max_len)
print(desc)
