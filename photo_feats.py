from pickle import load
import string

filename = 'FlickrText/Flickr8k.token.txt'

def load_doc(filename):
    with open(filename) as f:
        for line in f:
            text = f.read()
            return text
    f.close()

#doc = load_doc(filename)
#print(doc)

def load_description(filename):
    ls = dict()
    set = list()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        img_id, img_des = tokens[0], tokens[1:]
        img_id = img_id.split('.')[0]
        img_des = ' '.join(img_des)

        if img_id not in ls:
            ls[img_id] = list()

        ls[img_id].append(img_des)
    return ls

#desc = load_description(filename)
#print(len(doc2))

def clean_des(desc):
    tbl = str.maketrans('', '', string.punctuation)
    for key, desc_list in desc.items():
        for i in range(len(desc_list)):
            des = desc_list[i]
            des = des.split()
            des = [word.lower() for word in des]
            des = [w.translate(tbl) for w in des]
            des = [word for word in des if len(word)>1]
            des = [word for word in des if word.isalpha()]
            desc_list[i] = ' '.join(des)

#doc3 = clean_des(desc)
# print(doc3)

def to_vocab(desc):
    all_des = set()
    for key in desc.keys():
        [all_des.update(d.split()) for d in desc[key]]
    return all_des
#doc4 = to_vocab(desc)
#print(doc4)

def save_def(desc, filename):
    lines = list()
    for key, desc_list in desc.items():
        for des in desc_list:
            lines.append(key + ' ' + des)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


doc = load_doc(filename)
desc = load_description(doc)
print('Loaded: %d' % len(desc))

clean_des(desc)

vocab = to_vocab(desc)
print('Vocab size: %d' % len(vocab))
save_def(desc, 'descriptions_ted.txt')
