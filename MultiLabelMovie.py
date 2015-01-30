# coding=utf-8
__author__ = 'Jeremy'
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
from sklearn.cross_validation import train_test_split

from PrepareData import *


def createLB():
    f_labels = open("labels.txt","r").readlines()
    f_labels = [line.rstrip() for line in f_labels]
    lb = LabelEncoder()
    lb.fit(f_labels)
    return lb

def createMLB():
    labels_set = get_labels_set()
    mlb = MultiLabelBinarizer()
    mlb.fit(labels_set)
    return mlb

def load_movie_data():
    fr = open("labels_summary.txt","r")
    x_data, y_data = [], []
    lb = createLB()
    mlb = MultiLabelBinarizer()
    label_set = get_labels_set()
    for line in fr.readlines():
        line = line.rstrip()
        line_datas = line.split("--")
        summary = line_datas[-1]
        labels = line_datas[-2].split(' ')
        labels = [item for item in labels if item in label_set]
        if len(labels) == 0:
            continue
        labels = lb.transform(labels)
        x_data.append(summary)
        y_data.append(labels)
    y_data = mlb.fit_transform(y_data)
    return x_data, y_data, mlb, lb


def tfidf_transformer(x_train, y_train):
    countVec = CountVectorizer()
    transformer = TfidfTransformer(norm="l2")
    matrix = countVec.fit_transform(x_train)
    tfidf_matrix = transformer.fit_transform(matrix)
    return transformer, countVec, tfidf_matrix

def SVCClassifier(x_train, y_train):
    classifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
    return classifier

def statistic_result(y_test, y_score, lb):
    score_dict = {}
    n_class = y_score.shape[1]
    for i in range(n_class):
        precision = precision_score(y_test[:,i], y_score[:,i])
        recall = recall_score(y_test[:,i], y_score[:,i])
        f1_value = f1_score(y_test[:,i], y_score[:,i])
        label = lb.classes_[i]
        score_dict[label] = (precision, recall, f1_value)
    return score_dict


if __name__=="__main__":
    f=open("score.csv","w")
    x_data, y_data, mlb, lb = load_movie_data()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)
    tfidfTrans, countTrans, x_train = tfidf_transformer(x_train, y_train)
    x_test = tfidfTrans.transform(countTrans.transform(x_test))
    classifier = SVCClassifier(x_train, y_train)
    y_score = classifier.predict(x_test)
    score_dict = statistic_result(y_test, y_score, lb)
    for key,value in score_dict.items():
        line = key + ",%s,%s,%s" %  value
        f.write(line)
        f.write("\n")
    f.close()
