# Mohammed Murtuza Bhaiji
# 1001666586

import pandas as pd
import numpy as np
from sklearn import svm


def readData(file_name):
    df = pd.read_csv(file_name, sep=",", header=None)
    df = df.T
    labels = df.iloc[:, 0]
    data = df.iloc[:, 1:]
    return data, labels

def letter_2_digit_convert(mystr):
    mylist = []
    mystr = mystr.upper()
    for i in mystr:
        if i.isalpha():
            mylist.append(ord(i)-64)
    return mylist

def digit_2_letter_convert(mystr):
    mylist = []
    # mystr = mystr.upper()
    for i in mystr:
        mylist.append(chr(i+64))
    return mylist


def filterData(data,labels,filters):
    labels_lst = []
    data_lst = []
    labels_np = labels.to_numpy()
    for i in filters:
        indexes = labels_np==i
        data_lst.append(data.loc[indexes])
        labels_lst.append(labels.loc[indexes])
    return pd.concat(data_lst),pd.concat(labels_lst)

def svmClassifier(train, trainLabel, test, testLabel):
    SVM = svm.SVC(kernel='linear')
    predicted = SVM.fit(train, trainLabel)
    score = SVM.score(test, testLabel) * 100
    return testLabel, score

def splitData2TestTrain(data,labels,train_size,test_size):
    train_indexes = []
    test_indexes = []
    for i in range(data.shape[0]//(train_size+test_size)):
        train_indexes.extend(range(i*(train_size+test_size),i*(train_size+test_size)+train_size))
        test_indexes.extend(range(i*(train_size+test_size)+train_size,(i+1)*(train_size+test_size)))
    train_data = data.iloc[train_indexes]
    train_labels = labels.iloc[train_indexes]
    test_data = data.iloc[test_indexes]
    test_labels = labels.iloc[test_indexes]
    train_data = train_data.to_numpy()
    train_labels = train_labels.to_numpy()
    test_data = test_data.to_numpy()
    test_labels = test_labels.to_numpy()
    return train_data, train_labels,test_data, test_labels

if __name__ == "__main__":
    data, labels = readData("HandWrittenLetters.txt")
    filters = letter_2_digit_convert('MOEDBHJI')
    data, labels = filterData(data, labels, filters)

    x_train, y_train, x_test, y_test = splitData2TestTrain(data, labels, 29, 10)


    print("Train data Length:", len(x_train))
    print("Train Data:", x_train)

    print("Test length:", len(x_test))
    print("Test Data:", x_test)

    predictedClasses, svmScore = svmClassifier(x_train, y_train, x_test, y_test)

    print("Predicted classes: ", predictedClasses)
    print("Predicted class length", len(predictedClasses))


    trainFile = open("trainData.txt", "w")
    trainFile.writelines(["%s\n" % item for item in x_train])
    trainFile.close()

    testFile = open("testData.txt", "w")
    testFile.writelines(["\n" % item for item in x_test])
    testFile.close()

    classes = digit_2_letter_convert(predictedClasses)

    print("im converted classes: ", classes)

    labelsFile = open("predictedLabels.txt", "w")
    labelsFile.writelines(["%s\n" % item  for item in classes])
    labelsFile.close()

    print("Score for SVM: ", svmScore)





