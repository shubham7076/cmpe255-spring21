import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)

# To plot pretty figures

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGE_DIR = "images"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
    #path = "image" + ".png"
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

def random_digit(X):
    import matplotlib as mpl
    some_digit = X[36000]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,interpolation="nearest")
    plt.axis("off")

    save_fig("some_digit_plot")
    plt.show()
    return some_digit

   
def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml(name='mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
        #sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist["data"], mnist["target"]
    #return mnist


def sort_by_target(mnist_data,mnist_target):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist_data[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist_target[60000:])]))[:, 1]
    X_train = mnist_data[reorder_train]
    y_train = mnist_target[reorder_train]
    X_test = mnist_data[reorder_test + 60000]
    y_test = mnist_target[reorder_test + 60000]
    return X_train,y_train,X_test,y_test

def train_predict(X_train,y_train,some_digit):
    #import numpy as np
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # Example: Binary number 4 Classifier
    y_train_4 = (y_train == 4)
    #y_test_4 = (y_test == 4)

    from sklearn.linear_model import SGDClassifier
    # TODO
    # print prediction result of the given input some_digit
    sgd_clf = SGDClassifier(random_state=46)
    sgd_clf.fit(X_train, y_train_4)
    sgd_clf.predict([some_digit])

    #from sklearn.model_selection import cross_val_predict

    #y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_4, cv=3)

    #print("First 10 predictions of number 4: {0}".format(y_train_pred[:10]))

    calculate_cross_val_score(sgd_clf,X_train,y_train_4)
    #return y_train_pred
    
def calculate_cross_val_score(clf,X,y):
    # TODO
    from sklearn.model_selection import cross_val_score
    #clf = svm.SVC(kernel='linear', C=1, random_state=42)
    scores = cross_val_score(clf,X,y, cv=5,scoring="accuracy")
    print(f"Cross Validation Score: {scores}")

if __name__ == '__main__':

    mnist.data,mnist.target = load_and_sort()
    print(mnist_data)
    some_digit= random_digit(mnist_data)
    X_train,y_train,X_test,y_test = sort_by_target(mnist_data,mnist_target)
    #y_train_pred= train_predict(X_train,y_train,some_digit)
