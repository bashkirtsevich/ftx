from dataclasses import dataclass


@dataclass
class Node:
    encstate: int  # Encoder state of next node
    gamma: int  # Cumulative metric to this node
