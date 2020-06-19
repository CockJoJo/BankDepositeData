import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data(path):
    data = pd.read_csv(path, sep=";")
    return data


def feature_classifier(data):
    string_features = data.columns[data.dtypes == "object"].to_series().values
    int_features = data.columns[data.dtypes == "int64"].to_series().values
    float_features = data.columns[data.dtypes == "float64"].to_series().values
    numeric_features = np.append(int_features, float_features)

    bin_features = ['default', 'housing', 'loan', 'y']
    order_features = ['education']
    disorder_features = ['poutcome', 'job', 'marital', 'contact', 'month', 'day_of_week']

    return string_features, int_features, float_features, numeric_features, bin_features, order_features, disorder_features


def Missing_value_perprocessing_mean(train, test):
    col = train.columns
    imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(train)
    train = imp.fit(train)
    test = imp.fit(test)
    train = pd.DataFrame(train, columns=col)
    test = pd.DataFrame(test, columns=col)
    return train, test


def Missing_value_perprocessing_rf(train, test):
    Missing_features_dict = {}
    Missing_features_name = []
    # 先统计哪些列存在缺失的数据
    for feature in train.columns:
        Missing_count = train[train[feature].isnull()]['age'].count()
        if Missing_count > 0:
            # 统计包含缺失值的列
            Missing_features_dict.update({feature: Missing_count})
    # 对缺失的数据列按照缺失值数量从少到多排序，先拟合缺失值少的列
    Missing_features_name = sorted(Missing_features_dict.keys(), reverse=True)
    # print(Missing_features_name)
    for feature in Missing_features_name:
        # 训练集中有缺失值的数据
        train_miss_data = train[train[feature].isnull()]
        train_miss_data_X = train_miss_data.drop(Missing_features_name, axis=1)
        # 训练集中没有缺失值的数据
        train_full_data = train[train[feature].notnull()]
        train_full_data_Y = train_full_data[feature]
        train_full_data_X = train_full_data.drop(Missing_features_name, axis=1)
        # 测试集中有缺失值的数据
        test_miss_data = test[test[feature].isnull()]
        test_miss_data_X = test_miss_data.drop(Missing_features_name, axis=1)
        # 测试集中没有缺失值的数据
        test_full_data = test[test[feature].notnull()]
        test_full_data_Y = test_full_data[feature]
        test_full_data_X = test_full_data.drop(Missing_features_name, axis=1)
        from sklearn.ensemble import RandomForestClassifier
        # 使用随机森林拟合
        rf = RandomForestClassifier(n_estimators=100)
        # 利用训练集中没有缺失值的数据构建随机森林
        rf.fit(train_full_data_X, train_full_data_Y)
        # 预测训练集中的缺失值
        train_miss_data_Y = rf.predict(train_miss_data_X)
        # 预测测试集中的缺失值
        test_miss_data_Y = rf.predict(test_miss_data_X)
        # 将训练集中的缺失值补充完整
        train_miss_data[feature] = train_miss_data_Y
        # 将测试集中的缺失值补充完整
        test_miss_data[feature] = test_miss_data_Y
        # 将补充完整的
        train = pd.concat([train_full_data, train_miss_data])
        test = pd.concat([test_full_data, test_miss_data])
    return train, test


# 使用knn填补缺失值
def Missing_value_perprocessing_knn(bank_data_small_train, bank_data_small_test):
    Missing_features_dict = {}
    Missing_features_name = []
    # 先统计哪些列存在缺失的数据
    for feature in bank_data_small_train.columns:
        Missing_count = bank_data_small_train[bank_data_small_train[feature].isnull()]['age'].count()
        if Missing_count > 0:
            # 统计包含缺失值的列
            Missing_features_dict.update({feature: Missing_count})
    # 对缺失的数据列按照缺失值数量从少到多排序，先拟合缺失值少的列
    Missing_features_name = sorted(Missing_features_dict.keys(), reverse=True)
    from sklearn.neighbors import KNeighborsClassifier
    for feature in Missing_features_name:
        # 训练集中有缺失值的数据
        train_miss_data = bank_data_small_train[bank_data_small_train[feature].isnull()]
        train_miss_data_X = train_miss_data.drop(Missing_features_name, axis=1)
        # 训练集中没有缺失值的数据
        train_full_data = bank_data_small_train[bank_data_small_train[feature].notnull()]
        train_full_data_Y = train_full_data[feature]
        train_full_data_X = train_full_data.drop(Missing_features_name, axis=1)
        # 测试集中有缺失值的数据
        test_miss_data = bank_data_small_test[bank_data_small_test[feature].isnull()]
        test_miss_data_X = test_miss_data.drop(Missing_features_name, axis=1)
        # 测试集中没有缺失值的数据
        test_full_data = bank_data_small_test[bank_data_small_test[feature].notnull()]
        test_full_data_Y = test_full_data[feature]
        test_full_data_X = test_full_data.drop(Missing_features_name, axis=1)

        # 使用K近邻拟合
        knn = KNeighborsClassifier()
        forest = knn.fit(train_full_data_X, train_full_data_Y)

        train_miss_data_Y = knn.predict(train_miss_data_X)
        test_miss_data_Y = knn.predict(test_miss_data_X)

        train_miss_data[feature] = train_miss_data_Y
        test_miss_data[feature] = test_miss_data_Y

        bank_data_small_train = pd.concat([train_full_data, train_miss_data])
        bank_data_small_test = pd.concat([test_full_data, test_miss_data])

    return bank_data_small_train, bank_data_small_test


# 归一化
def Scale_perprocessing(Train):
    col = Train.columns
    copy = Train.copy()
    scaler = sklearn.preprocessing.MinMaxScaler()
    # 新版本中fit_transform的第一个参数必须为二维矩阵
    copy = scaler.fit_transform(copy)
    Train = pd.DataFrame(copy, columns=col)
    return Train


# 处理二分类的特征
def bin_features_perprocessing(bin_features, bank_data):
    for feature in bin_features:
        new = np.zeros(bank_data[feature].shape[0])
        for rol in range(bank_data[feature].shape[0]):
            if bank_data[feature][rol] == 'yes':
                new[rol] = 1
            elif bank_data[feature][rol] == 'no':
                new[rol] = 0
            else:
                new[rol] = None
        bank_data[feature] = new
    return bank_data


# 特征值没有次序的特征，一律使用onehot编码
def disorder_features_perprocessing(disorder_features, bank_data):
    for features in disorder_features:
        # 做onehot
        features_onehot = pd.get_dummies(bank_data[features])
        # 把名字改成features_values
        features_onehot = features_onehot.rename(columns=lambda x: features + '_' + str(x))
        # 拼接onehot得到的新features
        bank_data = pd.concat([bank_data, features_onehot], axis=1)
        # 删掉原来的feature columns
        bank_data = bank_data.drop(features, axis=1)
    return bank_data


# 特征值有次序关系的特征，按照特征值强弱排序（如：受教育程度）
def order_features_perprocessing(order_features, bank_data):
    education_values = ["illiterate", "basic.4y", "basic.6y", "basic.9y",
                        "high.school", "professional.course", "university.degree", "unknown"]
    replace_values = list(range(1, len(education_values)))
    replace_values.append(None)
    # 除了replace也可以用map()
    bank_data[order_features] = bank_data[order_features].replace(education_values, replace_values)
    bank_data[order_features] = bank_data[order_features].astype("float")
    return bank_data


# 学习曲线绘制方法
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


# 读取数据
def load_processeed_data(path):
    df = pd.read_csv(path)
    y = df.y
    df.drop('y', axis=1, inplace=True)
    x = df
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    return x_train, x_test, x, y_train, y_test, y, cv


# 随机森林
def rf_adjust_paraments(x_train, y_train, x_test, y_test, x, y):
    rf_score = []
    for i in range(0, 200, 5):
        rfc = RandomForestClassifier(n_estimators=i + 1, random_state=0, max_depth=10)
        score = cross_val_score(rfc, x, y, cv=10).mean()
        rf_score.append(score)
    print("max score: " + max(rf_score))
    plt.plot(range(1, 201, 5), rf_score)
    plt.show()

    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(x_train, y_train)

    # AbaBoost
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1, splitter='best'),
                             algorithm='SAMME',
                             n_estimators=200,
                             learning_rate=0.8)

    bdt.fit(x_train, y_train)
    y_pred = bdt.predict(x_test)
    print("混淆矩阵： " + sklearn.metrics.confusion_matrix(y_pred, y_test))
    print("训练集分数：", bdt.score(x_train, y_train))
    print("验证集分数", bdt.score(x_test, y_test))


def bdt_adjust_para(x, y, cv):
    bdt_score = []
    for i in range(0, 201, 5):
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i + 1, splitter='best'),
                                 algorithm='SAMME',
                                 n_estimators=200,
                                 learning_rate=0.8)
        score = cross_val_score(bdt, x, y, cv=10).mean()
        bdt_score.append(score)
    print("max score: " + max(bdt_score))
    plt.plot(range(1, 201, 5), bdt_score)
    plt.show()

    bdt_score = cross_val_score(bdt, x, y, cv=cv)
    print("AbaBoost 交叉验证最大分值：", bdt_score.max())
    print("AbaBoost 交叉验证平均分值：", bdt_score.mean())


# 学习曲线
def clf_learn_curve(clf, bdt, x, y, cv):
    estimator_Turple = (clf, bdt)
    title_Tuple = ("decision learning curve", "adaBoost learning curve")
    title = "decision learning curve"

    for i in range(2):
        estimator = estimator_Turple[i]
        title = title_Tuple[i]
        plot_learning_curve(estimator, title, x, y, cv=cv)
        plt.show()


if __name__ == '__main__':
    path = "bank-additional-full.csv"
    data = load_data(path)

    feature_classifier = feature_classifier(data)

    string_features = feature_classifier[0]
    # string_features, int_features, float_features, numeric_features, bin_features, order_features, disorder_features
    int_features = feature_classifier[1]
    float_features = feature_classifier[2]
    numeric_features = feature_classifier[3]
    bin_features = feature_classifier[4]
    order_features = feature_classifier[5]
    disorder_features = feature_classifier[6]

    data = bin_features_perprocessing(bin_features, data)
    data = order_features_perprocessing(order_features, data)
    data = disorder_features_perprocessing(disorder_features, data)

    # 打乱次序，划分训练集测试集
    # x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], train_size=0.8, random_state=0)

    data = data.sample(frac=1, random_state=12)

    x = data.iloc[0:round(data.shape[0] * 0.8), :]
    y = data.iloc[round(data.shape[0] * 0.8):, :]

    x, y = Missing_value_perprocessing_knn(x, y)

    x1_train = x.drop(['y'], axis=1).copy()
    y1_train = pd.DataFrame(x['y'], columns=['y'])

    x1_test = y.drop(['y'], axis=1).copy()
    y1_test = pd.DataFrame(y)
