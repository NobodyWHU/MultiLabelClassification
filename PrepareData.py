# coding=utf-8
__author__ = 'Jeremy'


def get_labels_set():
    fr = open("labels.txt","r")
    labels_set = set()

    for label in fr.readlines():
        label = label.rstrip()
        labels_set.add(label)

    fr.close()
    return labels_set


def print_label_result(mlb, lb, y_score):
    for label_set in mlb.inverse_transform(y_score):
        if len(label_set) == 0:
            print u"其他"
            continue
        label = lb.inverse_transform(label_set)
        for i in label:
            print i,
        print



