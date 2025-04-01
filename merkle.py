import hashlib
import networkx as nx
import matplotlib.pyplot as plt

class MerkleNode:
    def __init__(self, left=None, right=None, data=None):
        self.left = left
        self.right = right
        self.data = data
        self.hash = self.compute_hash()

    def compute_hash(self):
        if self.left is None and self.right is None:
            return hashlib.sha256(self.data.encode()).hexdigest()  # Leaf node hash
        else:
            combined_hash = self.left.hash + self.right.hash
            return hashlib.sha256(combined_hash.encode()).hexdigest()  # Internal node hash

class MerkleTree:
    def __init__(self, transactions):
        leaves = [MerkleNode(data=tx) for tx in transactions]
        self.root = self.build_tree(leaves)

    def build_tree(self, nodes):
        while len(nodes) > 1:
            temp = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]  # Handle odd nodes
                parent = MerkleNode(left, right)
                temp.append(parent)
            nodes = temp
        return nodes[0]

# Example transactions (synthetic data)
transactions = ["Tx1", "Tx2", "Tx3", "Tx4", "Tx5", "Tx6", "Tx7", "Tx8"]
merkle_tree = MerkleTree(transactions)

print("Merkle Root:", merkle_tree.root.hash)
