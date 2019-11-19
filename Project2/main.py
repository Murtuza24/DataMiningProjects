import pandas as pd
import  numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def filterData(data,labels,filters):
    labels_lst = []
    data_lst = []
    labels_np = labels.to_numpy()
    for i in filters:
        indexes = labels_np==i
        data_lst.append(data.loc[indexes])
        labels_lst.append(labels.loc[indexes])
    return pd.concat(data_lst),pd.concat(labels_lst)

def taskFunctios(task):

    if task.upper() == "A":

        data, labels = readData("HandWrittenLetters.txt")
        filters = letter_2_digit_convert('ABCDE')
        data, labels = filterData(data, labels, filters)
        #1 normal svm classifier
        x_train, y_train, x_test, y_test = splitData2TestTrain(data, labels, 30, 9)
        svmScore = svmClassifier(x_train, y_train, x_test, y_test)
        print("Task A score for SVM: ", svmScore)


        #2 KNN classifier
        # knnresult = kNearestNeighbor(x_train, y_train, x_test, 5)
        # print("kNN class result:", knnresult)
        # print("The accuracy for KNN is: ", printAccuracy(y_test,knnresult))
        knnscore = KnnSk(x_train, y_train, x_test, y_test, 3)
        print("\nThe accuracy for K-NN is: ", knnscore)

        #3 Linear Regression
        linRegScore = linearRegression(x_train.T,x_test.T,y_train.T,y_test.T)
        # linRegScore = lineraSK(x_train, y_train, x_test, y_test)
        print("\nThe accuracy for Linear Regression is: ", linRegScore)

        #4 Centroid Method
        centroidScore = centroid(x_train, y_train, x_test, y_test)
        print("\nThe accuracy for Centroid Method is: ", centroidScore)

    elif task.upper() == "B":
        data, labels = readData("ATNTFaceImages400.txt")
        svmScores = svmKFold(data, labels, 5)
        print("\n5 fold CV scores for SVM: ", svmScores)
        print("Total average score for 5 fold CV: ", np.mean(svmScores))

        knnScores = KnnKFold(data,labels, 5)
        print("\n5 fold CV scores for KNN: ", knnScores)
        print("Total average score for 5 fold CV: ", np.mean(knnScores))

        linRegScores =  linearRegressionKFold(data, labels, 5)
        print("\n5 fold CV scores for Linear Regression: ", linRegScores)
        print("Total average score for 5 fold CV: ", np.mean(linRegScores))

        centroidScores = centroidKFold(data, labels, 5)
        print("\n5 fold CV scores for Centroid Method: ", centroidScores)
        print("Total average score for 5 fold CV: ", np.mean(centroidScores))

    elif task.upper() == "C":
        data, labels = readData("HandWrittenLetters.txt")
        filters = letter_2_digit_convert('ABCDEFGHIJ')
        data, labels = filterData(data, labels, filters)

        splits = [[5, 34], [10, 29], [15, 24], [20, 19], [25, 24], [30, 9], [35, 4]]
        scores = []


        for split in splits:
            x_train, y_train, x_test, y_test = splitData2TestTrain(data, labels, split[0], split[1])
            scores.append(centroid(x_train, y_train, x_test, y_test))


        print("\nAccuracy scores for 7 Splits: ",scores)
        print("\nAverager accuracy %: ", np.mean(scores))
        result = scores
        X_axis = ["[5,34]", "[10,29]", "[15,24]", "[20,19]", "[25,24]", "[30,9]", "[35,4]"]
        plt.xlabel("Train, Test")
        plt.ylabel("Accuracy %")
        plt.plot(X_axis, result)

        plt.show()

    elif task.upper() == "D":
        data, labels = readData("HandWrittenLetters.txt")
        filters = letter_2_digit_convert('KLMNOPQRST')
        data, labels = filterData(data, labels, filters)

        splits = [[5, 34], [10, 29], [15, 24], [20, 19], [25, 24], [30, 9], [35, 4]]
        scores = []


        for split in splits:
            x_train, y_train, x_test, y_test = splitData2TestTrain(data, labels, split[0], split[1])
            scores.append(centroid(x_train, y_train, x_test, y_test))

        print("\nAccuracy scores for 7 Splits: ",scores)
        print("\nAverager accuracy %: ", np.mean(scores))

        result = scores
        X_axis = ["[5,34]", "[10,29]", "[15,24]", "[20,19]", "[25,24]", "[30,9]", "[35,4]"]
        plt.xlabel("Train, Test")
        plt.ylabel("Accuracy %")
        plt.plot(X_axis, result)

        plt.show()

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



'''
 Using the Scikit library to implement the svm method.
'''

def svmClassifier(train, trainLabel, test, testLabel):
    SVM = svm.SVC(kernel='linear')
    SVM.fit(train, trainLabel)
    return SVM.score(test,testLabel)*100


def svmKFold(data,labels,folds):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]
        scores.append(svmClassifier(train_data,train_labels,test_data,test_labels))
    return scores


'''
K Nearest Neighbors
'''

def KnnSk(train, trainLabel, test, testLabel, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train, trainLabel)
    knnscore = knn.score(test, testLabel)
    return  knnscore*100

def KnnKFold(data,labels,folds):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]
        scores.append(KnnSk(train_data,train_labels,test_data,test_labels, 3))
    return scores

'''
Linear regression classifier
'''

def lineraSK(Xtrain, Ytrain, Xtest, Ytest):
    linReg = LinearRegression()
    linReg.fit(Xtrain, Ytrain)
    linRegScore = linReg.score(Xtest, Ytest)
    return linRegScore*100

def linearRegression(Xtrain, Xtest, Ytrain, Ytest):
    RowToFill = 0
    A_train = np.ones((1, len(Xtrain[0])))
    A_test = np.ones((1, len(Xtest[0])))
    Xtrain_padding = np.row_stack((Xtrain, A_train))
    Xtest_padding = np.row_stack((Xtest, A_test))
    element, indx, count = np.unique(Ytrain, return_counts=True, return_index=True)
    element = Ytrain[np.sort(indx)]
    Ytrain_Indent = np.zeros((int(max(element)), count[0] * len(element)))
    for i, j in zip(count, element):
        Ytrain_Indent[int(j) - 1, RowToFill * i:RowToFill * i + i] = np.ones(i)
        RowToFill += 1
    B_padding = np.dot(np.linalg.pinv(Xtrain_padding.T), Ytrain_Indent.T)
    Ytest_padding = np.dot(B_padding.T, Xtest_padding)
    Ytest_padding_argmax = np.argmax(Ytest_padding, axis=0) + 1

    err_test_padding = Ytest - Ytest_padding_argmax
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    return TestingAccuracy_padding

def linearRegressionKFold(data,labels,folds):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]
        train_data = train_data.to_numpy()
        train_labels = train_labels.to_numpy()
        test_data = test_data.to_numpy()
        test_labels = test_labels.to_numpy()
        scores.append(linearRegression(train_data.T, test_data.T, train_labels.T, test_labels.T))
    return scores


'''
Centroid Method

'''
def centroid(trainVector, trainLabel, testVector, testLabel):
    trainVector = trainVector.transpose()
    trainLabel = trainLabel.transpose()
    testVector = testVector.transpose()
    testLabel = testLabel.transpose()


    jj = []
    result = []
    for j in range(0, len(trainVector[0]), 8):
        columnMean = []
        columnMean.append(trainLabel[j])
        for i in range(len(trainVector)):
            columnMean.append(np.mean(trainVector[i,j:j+7]))
        if not len(jj):
            jj = np.vstack(columnMean)
        else:
            jj = np.hstack((jj,(np.vstack(columnMean))))

    for iN in range(len(testVector[0])):
        distances = []
        for m in range(len(jj[0])):
            euclead = np.sqrt(np.sum(np.square(testVector[:,iN] - jj[1:, m])))
            distances.append([euclead,int(jj[0,m])])
            distances = sorted(distances, key=lambda distances: distances[0])
        result.append(distances[0][1])

    err_test_padding = testLabel - result
    TestingAccuracy_padding = (1 - np.nonzero(err_test_padding)[0].size / float(len(err_test_padding))) * 100
    return (TestingAccuracy_padding)

def centroidKFold(data,labels,folds):
    skf = StratifiedKFold(n_splits=folds)
    scores = []
    for train_index, test_index in skf.split(data, labels):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        train_labels = labels.iloc[train_index]
        test_labels = labels.iloc[test_index]

        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()
        train_labels = train_labels.to_numpy()
        test_labels = test_labels.to_numpy()


        scores.append(centroid(train_data,train_labels,test_data,test_labels))
    return scores

if __name__ == "__main__":
    task = input("Please enter the Task..(A,B,C,D): ")
    taskFunctios(task)