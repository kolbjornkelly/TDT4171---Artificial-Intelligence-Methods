from pickle import load

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Extract data:
data = load(open("sklearn-data.pickle", "rb"))
x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# Recode reviews:
hv = HashingVectorizer(n_features=2**4, binary=True)
X_train = hv.transform(x_train)
X_test = hv.transform(x_test)

# Train classifiers:
BernNB = BernoulliNB()
BernNB.fit(X=X_train, y=y_train)

DecTree = DecisionTreeClassifier()
DecTree.fit(X=X_train, y=y_train)

# Predict:
y_Bern_pred = BernNB.predict(X_test)
y_DecT_pred = DecTree.predict(X_test)

# Check accuracy:
print("Naive Bayes:", accuracy_score(y_test, y_Bern_pred))
print("Decision Tree:", accuracy_score(y_test, y_DecT_pred))
