from numpy import load, ubyte
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 決定木
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# インスタンス（ロジスティック識別）
clf1 = LogisticRegression()
clf1.fit(X,y)

# 正の大きな値の係数は正例の判定に関与、負の大きな値となる係数は負例に関与している
for f, w in zip(breast_cancer.feature_names, clf1.coef_[0]):
    print("{0:<23}: {1:6.2f}".format(f, w))

