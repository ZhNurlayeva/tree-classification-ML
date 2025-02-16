"""
decision_tree.py
=================
Implementation of a Decision Tree Classifier from scratch.

- Supports Gini, Entropy, and Misclassification as splitting criteria.
- Includes methods for tree training, prediction, and pruning.
- Uses single-feature binary tests at decision nodes.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np

# ========================
# 1. Define the Node Class
# ========================
class Node:
    """
    Represents a single node in a decision tree.

    Attributes:
    - feature_index (int): Index of the feature used for splitting.
    - threshold (float): Split threshold for the feature.
    - left_child (Node): Left subtree.
    - right_child (Node): Right subtree.
    - is_leaf (bool): Indicates whether the node is a leaf.
    - prediction (int): The class label if it's a leaf.
    - impurity (float): The impurity measure of the node.
    """

    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None,
                 is_leaf=False, prediction=None, impurity=0.0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.impurity = impurity if impurity is not None else 0.0  # Default impurity

    def decide(self, data_point):
        """
        Applies the decision rule to a given data point.

        Returns:
        - bool: True if data_point goes to the left child, False otherwise.
        """
        if self.is_leaf:
            return self.prediction
        return data_point[self.feature_index] <= self.threshold


# ===========================
# 2. Define the Decision Tree
# ===========================
class DecisionTree:
    """
    Implementation of a Decision Tree Classifier.

    Supports:
    - Gini, Entropy, and Misclassification for splitting.
    - Maximum depth and minimum impurity decrease for stopping.

    Methods:
    - fit(X, y): Trains the decision tree.
    - predict(X): Predicts labels for input data.
    """

    def __init__(self, max_depth=None, min_impurity_decrease=0.0, criterion="gini"):
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.root = None
        self.feature_importances_ = None

    def get_params(self, deep=True):
        """ Required for sklearn compatibility. """
        return {
            "max_depth": self.max_depth, 
            "min_impurity_decrease": self.min_impurity_decrease, 
            "criterion": self.criterion
        }

    def set_params(self, **params):
        """ Required for sklearn compatibility. """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    

    # ========================
    # 3. Training the Decision Tree
    # ========================
    def fit(self, X, y):
        """
        Trains the decision tree on the dataset.

        Parameters:
        - X (numpy array): Feature matrix.
        - y (numpy array): Target labels.
        """
        self.root = self._grow_tree(X, y, depth=0)
        self._compute_feature_importance(X)  # Compute feature importance after training

    def _grow_tree(self, X, y, depth):
        """
        Recursively grows the decision tree.

        Stopping Conditions:
        - Pure node (all samples belong to one class).
        - Maximum depth reached.
        - Impurity reduction below threshold.

        Returns:
        - Node: Root node of the constructed tree.
        """
        num_samples, num_features = X.shape
        unique_labels = np.unique(y)

        impurity = self._compute_impurity(y, y)

        # Stopping Criteria
        if len(unique_labels) == 1:
            return Node(is_leaf=True, prediction=unique_labels[0], impurity=impurity)
        if self.max_depth is not None and depth >= self.max_depth:
            majority_label = np.bincount(y).argmax()
            return Node(is_leaf=True, prediction=majority_label, impurity=impurity)

        # Find the best split
        best_feature, best_threshold, best_impurity, left_indices, right_indices = self._find_best_split(X, y)

        # Check impurity decrease condition
        if best_impurity < self.min_impurity_decrease:
            majority_label = np.bincount(y).argmax()
            return Node(is_leaf=True, prediction=majority_label, impurity=impurity)

        # Recursively grow left and right subtrees
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold,
                    left_child=left_child, right_child=right_child, impurity=impurity)
    
    # ========================
    # 4. Feature Importance Computation
    # ========================
    def _compute_feature_importance(self, X):
        """
        Computes feature importance based on split frequency.

        Importance is determined by counting how often a feature is used for splitting.
        """
        importance_scores = np.zeros(X.shape[1])

        def traverse(node):
            if node.is_leaf:
                return
            importance_scores[node.feature_index] += 1  # Count splits per feature
            traverse(node.left_child)
            traverse(node.right_child)

        traverse(self.root)
        if np.sum(importance_scores) > 0:
            self.feature_importances_ = importance_scores / np.sum(importance_scores)  # Normalize
        else:
            self.feature_importances_ = np.zeros_like(importance_scores)  # Handle edge case


    # ========================
    # 5. Finding the Best Split
    # ========================
    def _find_best_split(self, X, y):
        """
        Identifies the best feature and threshold for splitting.

        Returns:
        - best_feature (int): Index of the best splitting feature.
        - best_threshold (float): Best threshold for splitting.
        - best_impurity (float): Impurity measure for the best split.
        - left_indices, right_indices: Indices of the left/right split.
        """
        num_samples, num_features = X.shape
        best_feature, best_threshold, best_impurity = None, None, float("inf")
        left_indices, right_indices = None, None

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                impurity = self._compute_impurity(y[left_mask], y[right_mask])
                if impurity < best_impurity:
                    best_feature, best_threshold, best_impurity = feature, threshold, impurity
                    left_indices, right_indices = left_mask, right_mask

        return best_feature, best_threshold, best_impurity, left_indices, right_indices

    # ========================
    # 6. Impurity Computation
    # ========================
    def _compute_impurity(self, y_left, y_right):
        """
        Computes impurity for a given split.

        Parameters:
        - y_left (numpy array): Labels in the left split.
        - y_right (numpy array): Labels in the right split.

        Returns:
        - impurity (float): Computed impurity based on the chosen criterion.
        """
        num_left, num_right = len(y_left), len(y_right)
        num_total = num_left + num_right

        if num_left == 0 or num_right == 0:
            return float("inf")  # No valid split

        p_left = num_left / num_total
        p_right = num_right / num_total

        if self.criterion == "gini":
            return p_left * self._gini(y_left) + p_right * self._gini(y_right)
        elif self.criterion == "entropy":
            return p_left * self._entropy(y_left) + p_right * self._entropy(y_right)
        elif self.criterion == "misclassification":
            return p_left * self._misclassification(y_left) + p_right * self._misclassification(y_right)
        else:
            raise ValueError("Invalid criterion")

    # ==========================
    # 7. Impurity Metrics
    # ==========================
    def _gini(self, y):
        """Computes Gini impurity."""
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def _entropy(self, y):
        """Computes entropy impurity."""
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-9))  # Avoid log(0)

    def _misclassification(self, y):
        """Computes misclassification impurity."""
        proportions = np.bincount(y) / len(y)
        return 1 - np.max(proportions) 

    # ========================
    # 8. Prediction Functions
    # ========================
    def predict(self, X):
        """
        Predicts labels for new data points.

        Returns:
        - numpy array: Predicted labels.
        """
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """
        Predicts a single data point.
        """
        if node.is_leaf:
            return node.prediction
        return self._predict_single(x, node.left_child if x[node.feature_index] <= node.threshold else node.right_child)
