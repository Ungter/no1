from pickle import dump
from keras.preprocessing.text import Tokenizer

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

filename = 'FlickrText/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('dataset: %d' % len(train))

train_desc = load_clean_desc('descriptions_ted.txt', train)
print('desc: train=%d' % len(train_desc))

tokenizer = create_token(train_desc)

dump(tokenizer, open('token.pkl', 'wb'))