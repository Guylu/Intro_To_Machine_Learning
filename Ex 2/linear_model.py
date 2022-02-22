import numpy as np
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

PRICE_COL = 1


def fit_linear_regression(X, y):
    """
    The function returns two sets of values:
    the first is a numpy array of the coefficients vector ‘w‘ (think what
    should its dimension be)
    the second is a numpy array of the singular values of X.
    :param X:(numpy array with p rows and n columns
    :param y:numpy array with n rows
    :return:
    """
    singular_vals = np.linalg.svd(X, compute_uv=False)
    X_dagger_T = np.linalg.pinv(X)
    w = X_dagger_T.dot(y)
    return w, singular_vals


def predict(X, w):
    """
    :param X:a design matrix ‘X‘ (numpy array with p rows and m columns)
    :param w:coefficients vector
    :return:The function returns a numpy array with the predicted value by
    the model.
    """
    return X.dot(w)


def mse(res, pred):
    """
    calculated the mean squared error between 2 vectors
    :param res:
    :param pred:
    :return:
    """
    mean_s_error = 0
    m = res.shape[0]
    for i in range(m):
        mean_s_error += (res[i] - pred[i]) ** 2
    return mean_s_error / m


def load_data(path):
    """
    loads data from path, an filters it
    :param path: path of csv file
    :return: filtered numpy matrix
    """
    my_data = pd.read_csv(path)
    # delimiters to sort data out:
    delimiters = [my_data['price'] >= 0, my_data['bedrooms'] >= 0,
                  my_data['bathrooms'] >= 0,
                  my_data['sqft_living'] >= 0, my_data['sqft_lot'] >= 0,
                  my_data['floors'] >= 0,
                  my_data['waterfront'] >= 0, my_data['view'] >= 0,
                  my_data['condition'] >= 0,
                  my_data['grade'] >= 0, my_data['sqft_above'] >= 0,
                  my_data['sqft_basement'] >= 0, my_data['yr_renovated'] >= 0,
                  my_data['zipcode'] >= 0, my_data['sqft_living15'] >= 0,
                  my_data['sqft_lot15'] >= 0, my_data['sqft_living'] >= 0,
                  my_data['sqft_living'] >= 0]
    # apply delimiters:
    for deli in delimiters:
        my_data = my_data[deli]
    # dropping non essential data
    my_data.dropna(how='any', inplace=True)
    # rounding for easier dummies:
    my_data = my_data.drop('id', 1)
    my_data = my_data.round({'lat': 1, 'long': 1})
    # making dummies:
    my_data = pd.get_dummies(my_data, columns=['zipcode'])
    my_data = pd.get_dummies(my_data, columns=['lat'])
    my_data = pd.get_dummies(my_data, columns=['long'])

    # making sure we only have numbers in data
    my_data['date'] = my_data["date"].str.replace("T000000", "")
    return my_data


def plot_singular_values(sing_vals):
    """
    plots singular values
    :param sing_vals: array
    :return:nothing
    """
    # sort from large to small
    -np.sort(-sing_vals)
    # plot:
    plt.scatter(range(0, len(sing_vals)), sing_vals)
    plt.title("Q15 singular values of X ")
    plt.xlabel("index of singular value")
    plt.ylabel("singular value")
    plt.show()


def feature_evaluation(X, y):
    """
    This function, given the design matrix and response vector, plots for
    every non-categorical feature,
    a graph (scatter plot) of the feature values and the response values.
    It then also computes and shows on the graph the Pearson Correlation between
    the feature and the response.
    The graph’s title should include information about what feature is tested in
    that graph.
    :param X:
    :param y:
    :return:
    """
    X = X.drop('price', 1)
    # remove dummies:
    cols = [c for c in X.columns if c.lower()[:len("zipcode")] != 'zipcode']
    X = X[cols]
    cols = [c for c in X.columns if c.lower()[:len("lat")] != 'lat']
    X = X[cols]
    cols = [c for c in X.columns if c.lower()[:len("long")] != 'long']
    X = X[cols]
    # making sure everything is numbers:
    y = y.astype("float32")
    for column in X:
        col = X[column].values.astype("float32")
        pearson_correlation = np.cov(col, y, rowvar=False) / (
                np.std(col) * np.std(y))
        # getting the necessary info
        p = pearson_correlation[0][1]
        plt.scatter(col, y)
        plt.title("Q17 correlation between "
                  + str(column) + " and price\nwith ""Pearson Correlation "
                                  "of:" + str(p))
        plt.xlabel(column)
        plt.ylabel("price")
        plt.show()


# loading data:
X_dt = load_data(r"C:\University\Year 2\Semester 2\67577 Intro To Machine "
                 "Learning\Ex's\Ex 2\code\kc_house_data.csv")
# add ones row
X_dt = X_dt.append(pd.Series(1, index=X_dt.columns), ignore_index=True)
# making
# sure its all numbers:
X = X_dt.values.astype("float")
y = X[:, PRICE_COL]

result_all_data = fit_linear_regression(X, y)
plot_singular_values(result_all_data[1])
# making training data test data, anf the actual results for them
X = np.delete(X, PRICE_COL, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# num of element in the training data results
size = y_train.shape[0]
mse_nums = []
for p in range(1, 100):
    prec = round(size * (p / 100))
    # getting prediction w_hat
    w = fit_linear_regression(X_train[1:prec], y_train[1:prec])[0]
    pred = predict(X_test, w)
    # checking loss function
    b = mse(y_test, pred)
    print(str(p) + ": " + str(b))
    mse_nums.append(b)
plt.plot(mse_nums, label="MSE")
plt.title("Q16 - MSE of model by percentage\n of data being trained on")
plt.xlabel("Percentage of data being trained on")
plt.ylabel("Mean square error")
plt.legend()
plt.show()

# some times the graph slopes too quickly in the begining
# so here's the graph of the last 50 values
plt.plot(range(50, 100), mse_nums[49:], label="MSE")
plt.title("Q16 - MSE of model by percentage of data being trained "
          "on\npercentages 50-100")
plt.xlabel("Percentage of data being trained on")
plt.ylabel("Mean square error")
plt.legend()
plt.show()
# checking correlations:
feature_evaluation(X_dt, y)
