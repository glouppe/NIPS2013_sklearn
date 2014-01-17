# To save time, let us load the binary files generated above
import numpy as np

# Data loading
# ---------------

data = np.load("data/chr15-all-numpy.npz")
snps = data["snps"]
X = data["X"]
y = data["y"]

# X = X[:, :1000]

print "SNPs =", snps[:10]
print "X =", X
print "X.shape =", X.shape
print "y =", y

# Data impuration
# ---------------

print "Before imputation:", np.unique(X)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=-1, strategy="most_frequent")
X = imputer.fit_transform(X)

print "After imputation:", np.unique(X)


# Data visualization
# ------------------
# Project the data to a 2D space for visualization
from sklearn.decomposition import RandomizedPCA
Xp = RandomizedPCA(n_components=2, random_state=1).fit_transform(X)

# Setup matplotlib to work interactively
from matplotlib import pyplot as plt
import prettyplotlib as ppl

# Plot individuals
populations = np.unique(y)
# colors = plt.get_cmap("hsv")
f, ax = plt.subplots(figsize=(10, 4))

for i, p in enumerate(populations):
    mask = (y == p)
    ppl.scatter(ax, Xp[mask, 0], Xp[mask, 1], label=p)

plt.xlim([-50, 100])
ppl.legend(ax, loc=1)

plt.savefig("randomized_pca.png")


# Learn with scikit-learn
# -----------------------

from matplotlib import pyplot as plt


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



from sklearn.metrics import confusion_matrix

def print_cm(cm, labels):
    # Nicely print the confusion matrix
    print " " * 4,
    for label in labels:
        print " %s" % label,
    print

    for i, label1 in enumerate(labels):
        print label1,
        for j, label2 in enumerate(labels):
            print "%4d" % cm[i, j],
        print



from sklearn.linear_model import RidgeClassifierCV
clf = RidgeClassifierCV().fit(X_train, y_train)
print("Accuracy = ", clf.score(X_test, y_test)),

print_cm(confusion_matrix(y_test,
                          clf.predict(X_test),
                          labels=populations), populations)

# Plot coefficients
coef = np.mean(np.abs(clf.coef_), axis=0)

f, ax = plt.subplots(figsize=(10, 4))
plt.bar(left=range(coef.size), height=coef)
# ppl.bar(ax, left=range(coef.size), height=coef, xticklabels=None,
#             annotate=False)
plt.savefig("ridge.png")

# Top 10 SNPs
indices = np.argsort(coef)[::-1]

for i in range(10):
    print coef[indices[i]], snps[indices[i]]


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100,
                           max_features=0.2,
                           n_jobs=2,
                           random_state=1).fit(X_train, y_train)
print("Accuracy = ", clf.score(X_test, y_test)),
print_cm(confusion_matrix(y_test,
                          clf.predict(X_test),
                          labels=populations), populations)

# Plot importances
importances = clf.feature_importances_

f, ax = plt.subplots(figsize=(10, 4))
plt.bar(left=range(len(importances)), height=importances)
# ppl.bar(ax, left=range(coef.size), height=coef, xticklabels=None,
#         annotate=False)
plt.savefig("extra_trees.png")

# Top 10 SNPs
indices = np.argsort(importances)[::-1]

for i in range(10):
    print importances[indices[i]], snps[indices[i]]

print(clf.feature_importances_)
