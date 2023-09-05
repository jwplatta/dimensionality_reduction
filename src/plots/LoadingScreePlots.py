import matplotlib.pyplot as plt
import os

class LoadingScreePlots:
    def __init__(
      self,
      X,
      y,
      labels,
      components,
      explained_variance,
      explained_variance_ratio
    ):
        """
        Args:
            X (np.ndarray): data matrix
            y (np.ndarray): target vector
            labels (list): list of labels
            components (np.ndarray): components
            explained_variance (np.ndarray): explained variance
            explained_variance_ratio (np.ndarray): explained variance ratio
        """
        self.X = X # NOTE: only uses the first two features
        self.y = y
        self.labels = labels

        self.components = components
        self.explained_variance = explained_variance
        self.explained_variance_ratio = explained_variance_ratio
        self.fig = None


    def save(self, figsize=(20, 6), scree_scale=1.0, filename=None, directory="."):
        """
        Args:
            figsize (tuple): figure size
            scree_scale (float): scale the components by this value
            filename (str): name of the file to save
            directory (str): directory to save the file
        """
        if not filename:
            filename = "loading_scree_plots.png"

        try:
            self._build(figsize, scree_scale)
            self.fig.savefig(
              os.path.join(directory, filename),
              bbox_inches='tight',
              dpi=600
            )

            return filename
        except Exception as e:
            print(f"Error saving file: {e}")
            return False


    def show(self, figsize=(20, 6), scree_scale=1.0):
        """
        Args:
            figsize (tuple): figure size
            scree_scale (float): scale the components by this value
        """
        self._build(figsize, scree_scale)
        plt.show()


    def _build(self, figsize, scree_scale):
        self.fig, (self.axs_loading, self.axs_scree) = plt.subplots(
            nrows=1, ncols=2, figsize=figsize
        )
        self._build_loading(self.axs_loading)
        self._build_scree(self.axs_scree, scale=scree_scale)


    def _build_scree(self, axs, scale):
        colors = ['yellow', 'orange']
        for label in self.labels:
            axs.scatter(
                self.X[self.y == label][:, 0],
                self.X[self.y == label][:, 1],
                label=f'Class {label}',
                c=colors[label],
                marker='o',
                alpha=0.5
            )

        # NOTE: use the first two components
        for i, (comp, var) in enumerate(zip(
            self.components, self.explained_variance_ratio)):
            # NOTE: scale component by its variance explanation power
            comp = comp * var * scale
            axs.plot(
                [0, comp[0]],
                [0, comp[1]],
                label=f"Component {i + 1}",
                linewidth=2,
                color=f"C{i + 0}",
            )

        axs.set_title("Loading Plot")
        axs.set_xlabel("Feature A")
        axs.set_ylabel("Feature B")
        axs.legend()


    def _build_loading(self, axs):
        axs.bar(range(1, len(self.components) + 1), self.explained_variance)
        axs.set_title("Scree Plot")
        axs.set_xlabel("Component")
        axs.set_ylabel("Eigenvalue")