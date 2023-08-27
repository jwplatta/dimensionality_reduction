# Dimensionality Reduction

Repo contains simple implementations of some common dimensionality reduction techniques:
- Principal Component Analysis (PCA)

## PCA

PCA creates a new feature set by projecting the original data onto the eigenvectors of the covariance matrix of the original data.

The first principal component is calculated to account for the largest amount of variance the original data. The second principal component is calculated to account for the second largest amount of variance in the data and so on.

Each of these components respresents a vectors that explains the maximal amount of variance in the data. The more variance explained by the line, i.e. the larger the dispersion of the data points along that line, the more information represented by that line.


Each principal component is a linear combination of

These components are uncorrelated and the majority of the information in the original dataset is contained in the first few principal components. This approach allows PCA to capture more of the information in the original dataset while reducing the dimensionality of the dataset.

The eigenvectors of the covariance matrix of the original data are the vectors along which there is the most variance in the data.

- To compute the percentage of variance / information accounted for by each component, divide the eigenvalue by the sum of all the eigenvalues.

- You can simply keep all the components in order to describe the original data in terms of features that are uncorrelated.


Finally we use the features, i.e. eigenvectors from the covariance matrix, to project the original data onto a new feature space. To do this, we multiply the transpose

Pseudocode for algorithm:
1. Standardarize the data
2. Compute the covariance matrix
3. Calculate the eigenvectors and eigenvalues of the covariance matrix
4. Sort the eignevectors by eigenvalues in descending order
5. Select top n-components
6. Project the data into the selected components

### Examples

```sh
$ python main -a pca --data data.csv -n 2 -p 0.5 -s 42
```

