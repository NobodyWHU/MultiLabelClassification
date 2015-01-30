# coding:utf-8

__author__ = 'Jeremy'


def get_labels_set():
    fr = open("labels_summary.txt","r")
    labels_dict = {}
    for line in fr.readlines():
        line = line.rstrip()
        line_datas = line.split("--")
        labels = line_datas[-2].split(' ')
        for label in labels:
            if label not in labels_dict.keys():
                labels_dict.setdefault(label, 1)
            else:
                labels_dict[label] +=1

    for key, value in labels_dict.items():
        print key, value
    print "Done"

if __name__ == "__main__":
    get_labels_set()