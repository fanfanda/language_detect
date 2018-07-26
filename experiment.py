from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from utility import load_data, get_encoding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction import DictVectorizer

#测试四种语言的语种识别
corpus = ['chinese','english','german','japanese']
all_data = []

#载入数据
print("载入数据")
for label,language in enumerate(corpus):
    all_data += load_data(language,label)


all_data_x, all_data_y = zip(*all_data)



#计数的n元模型
print("生成CV")
myvectorizer = CountVectorizer(ngram_range = (1,2), max_features = 1000, analyzer = 'char_wb')
myvectorizer.fit(all_data_x)

#文档转换为向量
print("文档转换为向量")
all_data_vec_x = myvectorizer.transform(all_data_x)


#划分0.2的数据为测试集
print("划分训练测试集合")
x_train, x_test, y_train, y_test = train_test_split(all_data_vec_x, all_data_y, test_size = 0.2, random_state = 2018)

############################################ 朴素贝叶斯 ############################################
print("朴素贝叶斯")
classifier = MultinomialNB()
classifier.fit(x_train, y_train)

NB_result = classifier.predict(x_test)



############################################ lightgbm ############################################

# print("对于lightgbm,我们新加一个可能编码的特征")
# tuple_encoding_dict = tuple(map(lambda x: get_encoding(x), all_data_x))

# dv = DictVectorizer(sparse = False)
# tuple_encoding = dv.fit_transform(tuple_encoding_dict)


# all_data_vec_x = np.c_[all_data_vec_x.toarray(), tuple_encoding]

#这里的划分方式与朴素贝叶斯的相同，因为要做对比实验
# x_train, x_test, y_train, y_test = train_test_split(all_data_vec_x, all_data_y, test_size = 0.2, random_state = 2018)

clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 31, reg_alpha = 0.0, reg_lambda = 1,
        max_depth = -1, n_estimators = 2000, objective = 'softmax',
        subsample = 0.6, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.05, random_state = 2018, n_jobs = -1, silent = 1)
clf.fit(x_train.toarray(), y_train, eval_set = [(x_train.toarray() ,y_train),(x_test.toarray() , y_test)], eval_metric = 'multi_logloss',early_stopping_rounds = 200)
lgb_result = clf.predict(x_test.toarray())

print("------------ NaByes result -----------")
print('accuracy_score: ', accuracy_score(y_test, NB_result))
print('recall_score: ', recall_score(y_test, NB_result, average = 'micro'))
print('f1_score: ', f1_score(y_test, NB_result, average = 'weighted'))
print("------------ lgb result -----------")
print('accuracy_score: ', accuracy_score(y_test, lgb_result))
print('recall_score: ', recall_score(y_test, lgb_result, average = 'micro'))
print('f1_score: ', f1_score(y_test, lgb_result, average = 'weighted'))

print("------------ n-gram dict -----------")
print(myvectorizer.vocabulary_)
print("------------ 词文档的向量表示 -----------")
print(x_train.toarray())



