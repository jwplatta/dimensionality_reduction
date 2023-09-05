import os
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from src.models import PCA
from src.plots import LoadingScreePlots

# NOTE: Output filenames
ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
plots_filename = f"loading_scree_plots_{ts}.png"
results_filename = f"classification_reports_{ts}.txt"

# NOTE: generate data
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_repeated=1,
    n_classes=2,
    flip_y=0.2, # NOTE: add some noise
    # random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# NOTE: fit PCA on the training data until the n_components explains at
#      least 90% of the variance
pca = PCA(n_components=2)
pca.fit(X_train)

total_explained_variance = 0.0
n_components = 0
for idx, expl_var in enumerate(pca.explained_variance_ratio):
    total_explained_variance += expl_var
    print(f"total_explained_variance: {total_explained_variance} / {expl_var}")
    if expl_var < 0.1:
        n_components = idx + 1
        break

print(f"Explained variance ratios: {pca.explained_variance_ratio}")
print(
    f"n_components: {n_components} to explain {total_explained_variance} of the variance"
)

# NOTE: Apply PCA to the data for the number of components found above
pca = PCA(n_components=n_components)
pca.fit(X_train)

# NOTE: create a scree and loading plots
for ft1_idx in range(0, 9):
    for ft2_idx in range(0, 9):
        if ft1_idx == ft2_idx:
            continue
        else:
            plots = LoadingScreePlots(
                X_train[:, [ft1_idx, ft2_idx]],
                y_train,
                [0, 1],
                pca.components,
                pca.eigenvalues[:n_components],
                pca.explained_variance_ratio[:n_components]
            )
            filepath = plots.save(
              filename=f"loading_scree_plots_ft{ft1_idx}_ft{ft2_idx}.png",
              directory="./out",
              figsize=(20, 6),
              scree_scale=1.0,
            )
            print(f"Saved file: {filepath}")
            plt.close()


results_path = os.path.join("./out", results_filename)

# NOTE: train decision tree on the original data
dt_clf_X = DecisionTreeClassifier(max_depth=3)
dt_clf_X.fit(X_train, y_train)
y_pred_X = dt_clf_X.predict(X_test)

cr_X = classification_report(y_test, y_pred_X)
with open(results_path, "a") as f:
    f.write(cr_X)

# NOTE: transform the data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# NOTE: train decision tree on the transformed data
dt_clf_X_pca = DecisionTreeClassifier(max_depth=3)
dt_clf_X_pca.fit(X_train_pca, y_train)
y_pred_X_pca = dt_clf_X_pca.predict(X_test_pca)

cr_X_pca = classification_report(y_test, y_pred_X_pca)
with open(results_path, "a") as f:
    f.write(cr_X_pca)

