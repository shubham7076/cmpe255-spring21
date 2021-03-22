import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv("lab3\diabetes.csv", header=0, names=col_names, usecols=col_names)
        #print(self.pima.head())
        self.X_test = None
        self.y_test = None
        

    def define_feature(self,feature_cols,min_max):
        #feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        if(min_max == True):
            X=  (X - X.min())/(X.max() - X.min())
        y = self.pima.label
        return X, y
    
    def train(self,feature_cols,min_max):
        # split X and y into training and testing sets
        X, y = self.define_feature(feature_cols,min_max)
        X_train, self.X_test, y_train,self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self,feature_cols=['pregnant', 'insulin', 'bmi', 'age'],min_max = False):
        model = self.train(feature_cols,min_max)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)

    #def correlation(self):

    def plot(self,confusion_matrix):
        #plt.figure()
        plt.matshow(confusion_matrix,cmap= "Pastel2")

        for x in range(0,2):
            for y in range(0,2):
                plt.text(x,y,confusion_matrix[x,y])

        plt.ylabel("expected label")
        plt.xlabel("predicted label")
        plt.show()
    
        # [row, column]
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]

        print("Sensitivity: %.4f" % (TP / float(TP + FN)))
        print("Specificy  : %.4f" % (TN / float(TN + FP)))

if __name__ == "__main__":
    classifer = DiabetesClassifier()

    # Base solution
    result = classifer.predict()
    #print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    #print(f"score={score}")
    con_matrix_base = classifer.confusion_matrix(result)
    #print(f"confusion_matrix=${con_matrix}")
    print("Sensitivity and specificy value for Base_solution\n")
    plot = classifer.plot(con_matrix_base)
    #print(f"Plotting the confusion matrix {plot}")

    # Solution 1.0 

    columns = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age']
    solution_1 = classifer.predict(feature_cols=columns)
    score_1 = classifer.calculate_accuracy(solution_1)
    con_matrix_1 = classifer.confusion_matrix(solution_1)
    print("Sensitivity and specificy value for solution_1\n")
    plot = classifer.plot(con_matrix_1)


    # Solution 2.0
    featured_columns= ['pregnant', 'glucose','insulin','bmi','age','pedigree']
    solution_2 = classifer.predict(feature_cols=featured_columns)
    score_2 = classifer.calculate_accuracy(solution_2)
    con_matrix_2 =  classifer.confusion_matrix(solution_2)
    print("Sensitivity and specificy value for solution_2\n")
    plot = classifer.plot(con_matrix_2)

    # Solution 3.0

    solution_3 = classifer.predict(feature_cols= featured_columns,min_max=True)
    score_3 = classifer.calculate_accuracy(solution_3)
    con_matrix_3 =  classifer.confusion_matrix(solution_3)
    print("Sensitivity and specificy value for solution_3\n")
    plot = classifer.plot(con_matrix_3)

    print("| Experiement | Accuracy  | Confusion Matrix   | Comment                    |")
    print("|-------------|-----------|--------------------|----------------------------|")
    print("| Baseline    |",round(score,7),"|",*con_matrix_base,    "| Base|")
    print("| Solution 1  |",round(score_1,7),"|",*con_matrix_1,     "| All label features|")
    print("| Solution 2  |",round(score_2,7),"|",*con_matrix_2,     "| Most correlated Features|" )
    print("| Solution 3  |",round(score_3,7),"|",*con_matrix_3,     "| Most correlated features with Min Max Normalization|")



