from langdetect import detect,detect_langs
import chardet
from sklearn.feature_extraction import DictVectorizer


# v = DictVectorizer(sparse=False)
# D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
# X = v.fit_transform(D)
# print(v.transform({'foo': 4, 'baz': 3}))

def get_encoding(s):
    """
    获得可能的编码方式，原理是有限自动机
    """
    result = chardet.detect(s)
    return dict(set([(result['encoding'], result['confidence'])]))

# result = get_encoding(content.read())

def load_data(path, label):
    with open("dataset/" + path, 'rb') as content:
        lines = content.readlines()
    content_list = [(line.strip(), label) for line in lines]
    
    return content_list




# f = open('dataset/japan','r')
# fencoding = detect_langs(f.read())
# print(fencoding)