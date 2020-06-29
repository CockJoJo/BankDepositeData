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
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

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
def Missing_value_perprocessing_knn(train, test):
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
    from sklearn.neighbors import KNeighborsClassifier
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

        # 使用K近邻拟合
        knn = KNeighborsClassifier()
        forest = knn.fit(train_full_data_X, train_full_data_Y)

        train_miss_data_Y = knn.predict(train_miss_data_X)
        test_miss_data_Y = knn.predict(test_miss_data_X)

        train_miss_data[feature] = train_miss_data_Y
        test_miss_data[feature] = test_miss_data_Y

        train = pd.concat([train_full_data, train_miss_data])
        test = pd.concat([test_full_data, test_miss_data])

    return train, test

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
                        n_jobs=None, train_sizes=np.linspace(.2, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    print("step  1")
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

def bdt_adjust_para(x, y):
    bdt_score_n = []
    bdt_score_learn = []
    for i in range(1, 200, 20):
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, splitter='best'),
                                 algorithm='SAMME',
                                 n_estimators=i,
                                 learning_rate=0.8)
        score = cross_val_score(bdt, x, y, cv=5).mean()
        bdt_score_n.append(score)
        print("finish {:.2f}%".format(i / 200 * 100))
    print("max score: {:.4f}".format(max(bdt_score_n)))
    print("index is {}".format(bdt_score_n.index(max(bdt_score_n))))
    plt.plot(range(1, 200, 20), bdt_score_n)
    plt.show()

    for i in range(5, 15):
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, splitter='best'),
                                 algorithm='SAMME',
                                 n_estimators=bdt_score_n.index(max(bdt_score_n)) * 5,
                                 learning_rate=i / 10)
        score = cross_val_score(bdt, x, y, cv=5).mean()
        bdt_score_learn.append(score)
        print("finish {:.2f}%".format((i - 5) * 100))
    print("max score: {:.4f}".format(max(bdt_score_learn)))
    plt.plot(range(5, 15), bdt_score_learn)
    plt.show()

    print("AbaBoost n_estimater 测试交叉验证最大分值：", max(bdt_score_n))
    print("AbaBoost n_estimater 测试交叉验证平均分值：", bdt_score_n.mean())
    print("AbaBoost 交叉验证最大分值所选择n_estimate值为", bdt_score_n.index(max(bdt_score_n)) * 5)

    print("AbaBoost learning-rate 测试交叉验证最大分值：", max(bdt_score_learn))
    print("AbaBoost learning-rate 测试交叉验证平均分值：", bdt_score_learn.mean())
    print("AbaBoost 交叉验证最大分值所选择learning rate值为", (bdt_score_learn.index(max(bdt_score_learn)) / 10) + 0.5)

    return max(bdt_score_n), bdt_score_n.index(max(bdt_score_n)) * 20


# 学习曲线
def learn_curve(clf, rf, bdt, x, y, cv):
    estimator_Turple = (clf, rf, bdt)
    title_Tuple = ("Decision Tree learning curve", "Random Forest learning curve", "AdaBoost learning curve")

    for i in range(3):
        estimator = estimator_Turple[i]
        title = title_Tuple[i]
        plot_learning_curve(estimator, title, x, y, cv=cv)
        plt.show()



def clf_find_max_depth(x, y):
    clf_score = []
    for i in range(1, 50):
        clf = DecisionTreeClassifier(max_depth=i)
        score = cross_val_score(clf, x, y, cv=10).mean()
        clf_score.append(score)
        print("finish {:.2f}%".format(i / 50 * 100))
    plt.plot(range(1, 50), clf_score)
    plt.show()
    print("Maximum value in the score is " + str(max(clf_score)))
    print("The max depth of Maximum value in score is " + str(clf_score.index(max(clf_score))))
    return max(clf_score), clf_score.index(max(clf_score))

def svm_find_c(x, y):
    svm = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = sklearn.model_selection.GridSearchCV(svm, param_grid, n_jobs=8, verbose=1)
    grid_search.fit(x, y)
    best_parameters = grid_search.best_params_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    svm = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    svm.fit(x, y)
    return svm

def bdt_para(x, y):
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, splitter='best'), algorithm='SAMME')
    param_grid = {'n_estimater':[5,25,45,65,85,105,125,145,165,185,205],'learning_rate':[0.5,0.8,1.0,1.2,1.5]}
    grid_search = sklearn.model_selection.GridSearchCV(bdt,param_grid,n_jobs=8,verbose=1)
    grid_search.fit(x,y)
    best_parameters = grid_search.best_params_.get_params()
    for para,val in list(best_parameters.items()):
        print(para,val)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 14,splitter='best'),
                             algorithm='SAMME',
                             n_estimators=best_parameters['n_estimater'],
                             learning_rate=best_parameters['learning_rate'])
    bdt.fit(x,y)
    return bdt

def rf_find_n_estimate(x, y,maxdepth):
    rf_score = []
    for i in range(1, 200, 10):
        rf = RandomForestClassifier(n_estimators=i, max_depth=maxdepth)
        score = cross_val_score(rf, x, y, cv=5).mean()
        rf_score.append(score)
        print("rf finish {:.2f}%".format(i / 200 * 100))
    plt.plot(range(1, 200, 10), rf_score)
    plt.show()

    print("Maximum value in the score is " + str(max(rf_score)))
    print("The number of n_estimater who has best performance is " + str(rf_score.index(max(rf_score))*5))
    return max(rf_score), rf_score.index(max(rf_score))*5

def print_result_age(data):
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=40)
    plt.hist(data[data['y'] == 'yes']['age'], label="Yes", **kwargs)
    plt.hist(data[data['y'] == 'no']['age'], label="No", **kwargs)
    plt.legend()
    plt.show()


def print_job_result(print_data):
    job = print_data[print_data['y'] == 'yes'].groupby('job').count()['y'].index
    yes = print_data[print_data['y'] == 'yes'].groupby('job').count()['y']
    no = print_data[print_data['y'] == 'no'].groupby('job').count()['y']
    width = 0.5
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.bar(job, yes, width, color='yellow', label='Yes')
    plt.bar(job, no, width, bottom=yes, color='green', label='No')
    plt.xticks(rotation=90)
    plt.legend()
    plt.subplot(122)
    plt.bar(job, yes / no, width, color='blue', label='Yes/No')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def count_classifier(data):
    print("Yes:", data['y'][data['y'] == 'yes'].count())
    print("No:", data['y'][data['y'] == 'no'].count())  # 字符型属性各个属性值所占的比例
    for col in data.columns:
        if data[col].dtype == object:
            print(data.groupby(data[col]).apply(lambda x: x['y'][x['y'] == 'yes'].count() / x['y'].count()))

def duration_res_print(print_data):
    duration_count_yes = print_data[print_data['y'] == 'yes'].groupby('duration').count()['y']
    duration_count_no = print_data[print_data['y'] == 'no'].groupby('duration').count()['y']
    fig = plt.figure(figsize=(12, 6))

    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=30)
    plt.subplot(121)
    plt.title("Duration")
    plt.ylabel('Yes')
    plt.hist(duration_count_yes, label="Yes", **kwargs)
    plt.legend()
    plt.subplot(122)
    plt.title("Duration")
    plt.ylabel('No')
    plt.hist(duration_count_no, label="No", **kwargs)
    plt.legend()
    plt.show()

def edu_res_print(print_data):
    edu = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course",
           "university.degree", "unknown"]
    education_count_yes = print_data[print_data['y'] == 'yes'].groupby('education').count()['y']
    education_count_no = print_data[print_data['y'] == 'no'].groupby('education').count()['y']
    # reorder for edu
    education_count_yes = education_count_yes.reindex(index=edu)
    education_count_no = education_count_no.reindex(index=edu)
    y = education_count_yes
    n = education_count_no
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xticks(rotation=90)
    ax1.plot(y.values, 'b', label="Yes")
    ax1.set_xticks(np.arange(len(edu)))
    ax1.set_xticklabels(edu)
    ax1.set_ylabel('Yes')
    ax1.set_title("Education")
    plt.legend()
    # 加入第二根折线
    ax2 = ax1.twinx()
    ax2.plot(n.values, 'r', label="No")
    ax2.set_xticks(np.arange(len(edu)))
    ax2.set_xticklabels(edu)
    ax2.set_ylabel('No')
    plt.legend()
    plt.show()


def deposite_rate_print(print_data):
    edu = ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course",
           "university.degree", "unknown"]

    education_count_yes = print_data[print_data['y'] == 'yes'].groupby('education').count()['y']
    education_count_no = print_data[print_data['y'] == 'no'].groupby('education').count()['y']

    education_count_yes = education_count_yes.reindex(index=edu)
    education_count_no = education_count_no.reindex(index=edu)

    index = education_count_yes.index
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(1, 1, 1)
    axes.plot((education_count_yes / (education_count_yes + education_count_no)).values, label='Yes/(Yes+No)')
    axes.set_xticks(np.arange(len(edu)))
    axes.set_xticklabels(edu)
    axes.set_title("Education")
    axes.set_ylabel('Yes/(Yes+No)')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


def pie_res(print_data, feature):
    index = print_data.groupby(feature).count().index
    yes = print_data[print_data['y'] == 'yes'].groupby(feature).count()['y']
    no = print_data[print_data['y'] == 'no'].groupby(feature).count()['y']

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title("YES")
    plt.pie(yes.values, labels=index)
    plt.subplot(122)
    plt.title("NO")
    plt.pie(no.values, labels=index)
    plt.show()

def SMOTE_unbalance(feature, result):
    sample_solver = SMOTE(random_state=0)
    feature_sample, result_sample = sample_solver.fit_sample(feature, result)
    return feature_sample, result_sample

def count_nan(data):
    for col in data.columns:
        if data[col].dtype == object:
            print("Percentage of \"unknown\" in %s：" % col,
                  data[data[col] == "unknown"][col].count(), "/", data[col].count())

if __name__ == '__main__':
    path = "bank-additional-full.csv"
    data = load_data(path)

    features_classifier = feature_classifier(data)

    string_features = features_classifier[0]
    # string_features, int_features, float_features, numeric_features, bin_features, order_features, disorder_features
    int_features = features_classifier[1]
    float_features = features_classifier[2]
    numeric_features = features_classifier[3]
    bin_features = features_classifier[4]
    order_features = features_classifier[5]
    disorder_features = features_classifier[6]

    data = bin_features_perprocessing(bin_features, data)
    data = order_features_perprocessing(order_features, data)
    data = disorder_features_perprocessing(disorder_features, data)

    # 打乱次序，划分训练集测试集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], train_size=0.8,
                                                        random_state=0)

    data = data.sample(frac=1, random_state=12)

    x = data.iloc[0:round(data.shape[0] * 0.8), :]
    y = data.iloc[round(data.shape[0] * 0.8):, :]

    x, y = Missing_value_perprocessing_knn(x, y)

    x1_train = x.drop(['y'], axis=1).copy()
    y1_train = pd.DataFrame(x['y'], columns=['y'])

    x1_test = y.drop(['y'], axis=1).copy()
    y1_test = y1_test = pd.DataFrame(y.y)

    feature = pd.concat([x1_train, x1_test], axis=0, ignore_index=True)
    result = pd.concat([y1_train, y1_test], axis=0, ignore_index=True)
    feature_sampled = SMOTE_unbalance(feature, result)[0]
    result_sampled = SMOTE_unbalance(feature, result)[1]
    feature_sampled = feature_sampled.drop('campaign',axis=1)
    feature_sampled_train,feature_sampled_test,result_sampled_train,result_sampled_test = train_test_split(feature_sampled,result_sampled,test_size=0.2,random_state=7)

    clf_max_depth = clf_find_max_depth(feature_sampled_train, result_sampled_train)
    clf = DecisionTreeClassifier(max_depth=clf_max_depth[1])
    clf.fit(feature_sampled_train,result_sampled_train)
    clf_train_score = clf.score(feature_sampled_train,result_sampled_train)
    clf_predict = clf.predict(feature_sampled_test)
    clf_confusion_metrix = sklearn.metrics.confusion_matrix(result_sampled_test,clf_predict)
    clf_test_score=clf.score(feature_sampled_test,result_sampled_test)

    rf_bst_para = rf_find_n_estimate(feature_sampled_train, result_sampled_train,clf_max_depth[1])
    rfc = RandomForestClassifier(max_depth=clf_max_depth[1],n_estimators=rf_bst_para[1])
    rfc.fit(feature_sampled_train,result_sampled_train)
    rfc_train_score = rfc.score(feature_sampled_train,result_sampled_train)
    rfc_predic = rfc.predict(feature_sampled_test)
    rfc_confusion_metrix = sklearn.metrics.confusion_matrix(result_sampled_test,rfc_predic)
    rfc_test_score = rfc.score(feature_sampled_test,result_sampled_test)

    bdt_best_para = bdt_adjust_para(feature_sampled_train,result_sampled_train)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=clf_max_depth[1], splitter='best'),
                             algorithm='SAMME',
                             n_estimators=bdt_best_para[1])
    #报错，需要将result_sampled_train转化为一维数组
    bdt.fit(feature_sampled_train,result_sampled_train)
    bdt_train_score = bdt.score(feature_sampled_train,result_sampled_train)
    bdt_predict = bdt.predict(feature_sampled_test)
    bdt_confusion_metrix = sklearn.metrics.confusion_matrix(result_sampled_test,bdt_predict)
    bdt_test_score = bdt.score(feature_sampled_test,result_sampled_test)

    cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

    learn_curve(clf, rfc, bdt, feature_sampled, result_sampled, cv)
