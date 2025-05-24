import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.data_pointer = 0
        self.size = 0

    def add(self, priority, data):
        """
        Adds a new experience and its priority to the SumTree.
        If the buffer is full, it overwrites the oldest experience.
        """
        tree_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        """
        Updates the priority of an existing experience.
        Propagates the change up the tree.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        """
        Retrieves an experience from the tree based on a random value.
        Used for sampling.
        """
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def max_priority(self):
        return (
            self.tree[self.capacity - 1 : 2 * self.capacity - 1].max()
            if self.size > 0
            else 1.0
        )
