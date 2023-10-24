##########################################################################
# Business Problem
# It is desired to develop a machine learning model that can predict
# whether people have diabetes when their characteristics are specified.
# You are expected to perform the necessary data analysis and
# feature engineering steps before developing the model.
###########################################################################
# Data Story
# The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# The research is conducted on the "diabetes study" conducted for "diabetes";
# 1 indicates positive diabetes test result,
# 0 indicates negative diabetes test result.
##########################################################################
# Pregnancies   : Number of pregnancies
# Glucose       : 2-hour plasma glucose concentration in the oral glucose tolerance test
# BloodPressure : Blood Pressure (Smallness) (mm Hg)
# SkinThickness : Skin Thickness
# Insulin       : 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction : Function (2-hour plasma glucose concentration in the oral glucose tolerance test)
# BMI           : body mass index
# Age           : Age  (year)
# Outcome       : have the disease (1) or not (0)
#############################################################################
# DUTY 1 : EDA
# Stage 1: Genel resmi inceleyiniz.
# Stage 2: Numerikvekategorikdeğişkenleriyakalayınız.
# Stage 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# Stage 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
# Stage 5: Aykırı gözlem analizi yapınız.
# Stage 6: Eksik gözlem analizi yapınız.
# Stage 7: Korelasyon analizi yapınız.
###########################################################################
# imports
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    # !pip install missingno
    import missingno as msno
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
    pd.set_option('display.width', 500)
# read data

##############################################################################
"""Stage 1 : Genel resmi inceleyiniz """
##############################################################################
run_settings()
df = read_data()
df_= df.copy()
df.columns = [col.lower() for col in df.columns]
xsamplenumber = df.shape[0]

# for test df.shape  (768,9)
# check_df(df,xplot=True)

## Max değerler incelendiğinde outlier değerler olduğunu düşünüyorum.
# Tipler incelendiğinde tüm veri sayısal olarak tutulmuş.
"""
# Stage 2 : Numerikvekategorikdeğişkenleriyakalayınız.
# Stage 3 : Numerik ve kategorik değişkenlerin analizini yapınız. """
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Unique value sayılarını incelediğimde ise tüm sayısal değişkenlerin sınıf sayılarının 10 üzerinde yani çok çeşitli olduğu görülüyor.
"""
# Stage 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
"""
for col in num_cols:
    print(f"############### {col}/Outcome Analyse #######################")
    print(df.groupby(col)["outcome"].agg(["mean",lambda x: x.count() / xsamplenumber]),end="\n")

""" Stage 5: Aykırı gözlem analizi yapınız. """
outlier_analyser(df,num_cols)

""" Stage 6: Eksik gözlem analizi yapınız."""
missing_value_analyser(df,num_cols)

""" Stage 7: Korelasyon analizi yapınız."""
print(f"##################### Corelation Analyse #######################")
corr= df[df.columns].corr()
print(corr["outcome"].T.sort_values(ascending=False))

sns.set(rc={"figure.figsize":(20,20)})
sns.heatmap(corr,cmap="RdBu")
plt.show(block=True)
############################################################################
# DUTY 2: Feature Engineering
# Stage 1:  Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0 olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN
# olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
# Stage 2: Yeni değişkenler oluşturunuz.
# Stage 3:  Encoding işlemlerini gerçekleştiriniz.
# Stage 4: Numerik değişkenler için standartlaştırma yapınız.
# Stage 5: Model oluşturunuz.
############################################################################
""" Stage 1 """
df_cols_except_target = [col for col in df.columns if "outcome" not in col]

#  eksik değerleri ortalama değerler ile dolduracağım için öncelikle aykırı değerleri düzenliyorum
#  yoksa ortamlamalarım da etkilenecek.

### handle outlier values
outlier_analyser(df,df_cols_except_target)
for col in df_cols_except_target:
    replace_with_thresholds(df,col)

### handle zero values
print("########################### 0 Value Analyse ######################## ")
for col in df_cols_except_target:
    print( f"{col} : {df.loc[df[col] == 0].shape[0]}" )

### pregnancies bu çalışmanın dışında kalacak. Insülin 374 ve SkinThickness 227 veri 0
having_zero_value_cols = [col for col in df_cols_except_target if df.loc[df[col] == 0].shape[0] > 0]
work_zero_value_cols = [col for col in having_zero_value_cols if col not in ['pregnancies']]
## replace NAN values for zero values.
for col in work_zero_value_cols:
    df.loc[df[col] == 0, col] = None

### handle missing values
## yaş kırılımını inceleyip yeni bir değişkende sınıflandırma yapmak istiyorum.
# df["age"].describe(np.arange(0,1,0.1)).T
df["nw_age_cat"] = pd.cut(df["age"],bins=(0,21,23,25,27,29,33,38,43,50,70,100))

for col in work_zero_value_cols:
    df[col].fillna(df.groupby(["nw_age_cat","outcome"])[col].transform("median"),inplace=True)

missing_value_analyser(df,num_cols)

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
## mynotes :
    # one_hot_encode median verisi ile accuracy 0.8614718614718615
    # one_hot_encode mean verisi ile accuracy 0.8484848484848485

plot_importance(rf_model, X_train,len(X))
## result : insulin,glucose,skintickness,..


