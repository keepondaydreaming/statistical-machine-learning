"""
Name: Nguyen Duy Quang
Class: K28
MSSV: 21025014

You should understand your code
"""

from typing import Iterable

import numpy as np
import pandas as pd


def compute_entropy(freq: np.ndarray) -> float:
    probability = freq / np.sum(freq)
    return np.sum(-1 * probability * np.log2(probability))


def compute_feature_entropy(column: pd.DataFrame, y: pd.DataFrame, value: str) -> float:
    indices = column.index[column == value].values
    filtered_y = y.iloc[indices]
    freq = filtered_y.value_counts().values
    feature_entropy = len(indices) / len(column) * compute_entropy(freq)

    return feature_entropy


def get_data_idx(column: pd.DataFrame, value: str) -> np.ndarray:
    indices = column.index[column == value]
    return indices.values


# Q2.1
def compute_information_gain(X: pd.DataFrame, y: pd.DataFrame, feature: str) -> float:
    values = X[feature].unique().tolist()

    avg_feature_entropy = 0
    for value in values:
        avg_feature_entropy += compute_feature_entropy(X[feature], y, value)

    entropy = compute_entropy(y.value_counts().values)

    return entropy - avg_feature_entropy


# Build decision tree on X and y
# List of:
# node_index, node_feature[0..3], (feature_value -> child_index) : internal node
# leafnode: node_index, node_features = -1, Yes/No


class Node:
    def __init__(self, **kwargs):
        self.depth: int = kwargs.get("depth")
        self.value: str = kwargs.get("value")
        self.feature: str = kwargs.get("feature")
        self.entropy: float = kwargs.get("entropy")
        self.children: list[Node] = kwargs.get("children")
        self.data_idx: np.ndarray = kwargs.get("data_idx")
        self.target: str = kwargs.get("target")


class ID3Tree:
    def __init__(self):
        self.root: Node = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        self.features: Iterable[str] = X.columns
        self.target: Iterable[str] = y.columns

        self.root = Node(
            depth=0,
            entropy=compute_entropy(y.value_counts().values),
            data_idx=y.index.values,
        )

        stack = [self.root]
        while stack:
            node = stack.pop()

            if node.entropy > 0.0:
                node.children = self._split(X, y, node)
                stack += node.children
            else:
                self._set_leaf(y, node)

    def traverse(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.children:
                stack += node.children

            print(node.depth, node.feature, node.value, node.target)

    def _split(self, X: pd.DataFrame, y: pd.DataFrame, node: Node) -> list[Node]:
        X = X.iloc[node.data_idx]
        best_feature = self._find_best_feature(X, y)
        node.feature = best_feature

        values = X[best_feature].unique().tolist()
        children = [
            Node(
                value=value,
                depth=node.depth + 1,
                entropy=compute_feature_entropy(X[best_feature], y, value),
                data_idx=get_data_idx(X[best_feature], value),
            )
            for value in values
        ]

        return children

    def _set_leaf(self, y: pd.DataFrame, node: Node):
        targets = y.iloc[node.data_idx].mode()
        node.target = targets[self.target[0]][0]

    def _find_best_feature(self, X: pd.DataFrame, y: pd.DataFrame) -> str:
        best_gain = 0
        best_feature = None

        for feature in self.features:
            gain = compute_information_gain(X, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        return best_feature

    def _predict_one(self, data: pd.DataFrame) -> str:
        node = self.root
        for i in range(5):
            if node.target:
                return node.target
            
            for child in node.children:
                if data[node.feature] == child.value:
                    node = child
                    break


    def predict(self, data: pd.DataFrame) -> list[str]:
        results = []
        for idx in range(len(data)):
            prediction = self._predict_one(data.iloc[idx])
            results.append(prediction)
        return results


if __name__ == "__main__":
    df = pd.read_csv("./train.csv")
    features = ["Outlook", "Temperature", "Humidity", "Wind"]
    target = ["PlayTennis"]

    X = df[features]
    y = df[target]

    # Q2.1
    print("Information Gain of Outlook: ", compute_information_gain(X, y, features[0]))

    # Q2.2
    tree = ID3Tree()
    tree.fit(X, y)
    #tree.traverse()

    #
    df_test = pd.read_csv('test.csv')
    X_test = df_test[features]
    y_test = df_test[target].values.reshape(-1).tolist()

    preds = tree.predict(X_test)
    print("ground truth: ", y_test)
    print("prediction: ", preds)
    
    count = 0
    for gt, pred in zip(y_test, preds):
        if gt == pred:
            count += 1
    print("acc ", count / len(y_test))

