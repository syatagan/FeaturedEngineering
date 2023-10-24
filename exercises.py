import seaborn as sns
import pandas as pd
pd.set_option("display_max_column",None)
df = sns.load_dataset("titanic")

df["age"].isnull().sum()
df.isnull().sum().sort_values(ascending=False)
## bütün sütunlara bak ve satır bazında all değerlendirmesi yap.
df.notnull().all(axis=1)
## bütün satırlara bak ve sütun bazında all değerlendirmesi yap.
df.notnull().all(axis=0)

## bütün sütunlara bak ve satır bazında any değerlendirmesi yap.
df.isnull().any(axis=1)
## bütün satırlara bak ve sütun bazında any değerlendirmesi yap.
df.isnull().any(axis=0)


df["age"].fillna(df.groupby("sex")["age"].transform("median")).isnull().sum()
df["age"].isnull().sum()
df[df.index==1].T

msno.bar()
plt.show(block=True)
msno.matrix()
plt.show(block=True)
msno.heatmap()
plt.show(block=True)
