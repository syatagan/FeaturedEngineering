    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    # !pip install missingno
    import missingno as msno
    import random
    import warnings
    import lime
    from datetime import date
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
    from sklearn.utils import stats
    from Src.utils import grab_col_names
    from Src.utils import check_df
    from Src.utils import outlier_thresholds
    from Src.utils import check_outlier
    from Src.utils import replace_with_thresholds
    from Src.utils import check_MissingValue
from lime import lime_tabular
    from scipy import stats # Korelasyon testi için
# Configurations
def run_settings():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    pd.set_option('display.width', 500)
def read_data():
    xdf= pd.read_csv("Datasets/diabetes.csv")
    return xdf

def outlier_analyser(xdf, xcol_list) :
    print(f"##################### Outlier Analyse #######################")
    for col in xcol_list:
        print(f" {col} : {check_outlier(xdf, col)}")

def missing_value_analyser(xdf, xcol_list):
    print(f"##################### Missing Value Analyse #######################")
    for col in xcol_list:
        print(f" {col} : {check_MissingValue(xdf, col)}")

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def plot_importance(model, features, num=10, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
# read data

df = read_data()
df_ = df.copy()
df.columns = [col.lower() for col in df.columns]
xsamplenumber = df.shape[0]
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df_cols_except_target = [col for col in df.columns if "outcome" not in col]

### outliers insulin, bmi
# df[["insulin","skinthickness",'bmi','glucose']].describe().T
####

for col in df_cols_except_target:
    replace_with_thresholds(df,col)

having_zero_value_cols = [col for col in df_cols_except_target if df.loc[df[col] == 0].shape[0] > 0]
work_zero_value_cols = [col for col in having_zero_value_cols if col not in ['pregnancies']]
## replace NAN values for zero values.
for col in work_zero_value_cols:
    df.loc[df[col] == 0, col] = None

### handle missing values
## yaş kırılımını inceleyip yeni bir değişkende sınıflandırma yapmak istiyorum.
# df["age"].describe(np.arange(0,1,0.1)).T
df["nw_age_cat"] = pd.cut(df["age"],bins=(0,21,23,25,27,29,33,38,43,50,70,100))
df["nw_age_cat_diabetes"] = df.apply(lambda x: str(x["nw_age_cat"]) + "_" + str(x["outcome"]),axis=1)
# df["nw_age_cat"].value_counts()
for col in work_zero_value_cols:
    df[col].fillna(df.groupby(["nw_age_cat_diabetes"])[col].transform("median"),inplace=True)

## boş verileri doldurduktan sonra yeni değişkenler
df["nw_glucose_cat"] = pd.cut(df["glucose"],bins=(0,140,200,1000),labels=["Normal","Impaired","High"])
df["nw_insulin_cat"] = pd.cut(df["insulin"],bins=(0,16,166,1000),labels=["Low","Normal","High"])
df["nw_bmi_cat"] = pd.cut(df["bmi"],bins=(0,18.50,25,1000),labels=["Underweight","Normal","Overweight"])

ohe_cols = [col for col in df.columns if "nw_" in col]
for col in ohe_cols:
    df = one_hot_encoder(df,ohe_cols, drop_first=True)

""" sadece görmek için denedim bu problem için label encoding uygun değil. 
le_cols = [col for col in df.columns if "nw_" in col]
le = LabelEncoder()
for col in le_cols:
    df[col] = le.fit_transform(df[col])
"""
rs = RobustScaler()
for col in num_cols:
    df[col] = rs.fit_transform(df[[col]])

"""# Stage 5: Model oluşturunuz."""
y = df["outcome"]
X = df.drop(["outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)
y_pred y_test

explainer = lime_tabular.LimeTabularExplainer(X_train, mode="regression", feature_names= boston.feature_names)
explainer
idx = random.randint(1, len(X_test))

print("Prediction : ", rf_model.predict(X_test[idx].reshape(1,-1)))
print("Actual :     ", Y_test[idx])

explanation = explainer.explain_instance(X_test[idx], lr.predict, num_features=len(boston.feature_names))
## mynotes :
    # one_hot_encode median verisi ile accuracy 0.8614718614718615
    # one_hot_encode mean verisi ile accuracy 0.8484848484848485
    # one hot eksik verileri sadece outcome verisinden median aldığımda accuracy : 0.8701298701298701
    # one hot eksik veriler için yaş aralığı ve outcome olma durumlarını birleştirip yeni bir özellik atadım ve median verisi ile doldurdum.
    # Bu durumda model accuracy : 9653679653679653
plot_importance(rf_model, X_train,len(X))
## result : insulin,glucose,skintickness,..

df.head()
