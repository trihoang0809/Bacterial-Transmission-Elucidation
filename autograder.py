# Tri Hoang
# th52
# COMP 182 Spring 2023 - Homework 5, Problem 3

# You may NOT import anything apart from already imported libraries.
# You can use helper functions from provided.py, but they have
# to be copied over here.

from typing import Tuple
from collections import *
from copy import *


def reverse_digraph_representation(graph: dict) -> dict:
    """
    This function takes as input a weighted digraph graph in the standard representation
    and returns exactly the same weighted digraph graph but in the reversed representation.

    Input: graph: a weighted digraph graph in the standard representation

    Output: reversed_graph: the same weighted digraph graph but in the reversed representation.
    """
    reversed_graph = {node: {} for node in graph.keys()}
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            reversed_graph[neighbor][node] = weight
    return reversed_graph


# g0 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
#       4: {1: 4}, 5: {}}
# print(reverse_digraph_representation(g0))
# actual = expected = {0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {0: 20, 2: 8}, 4: {2: 20, 3: 4}, 5: {1: 16, 3: 8}}

# g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {
#     3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# print(reverse_digraph_representation(g0))
# actual = expected = {0: {}, 1: {0: 2, 4: 2}, 2: {0: 2, 1: 2}, 3: {0: 2, 2: 2}, 4: {2: 2, 3: 2}, 5: {1: 2, 3: 2}}


def modify_edge_weights(rgraph: dict, root: int) -> None:
    """
    This function takes as input a weighted digraph graph in the reversed representation
    and a node root, and modifies the edge weights of graph according to Lemma 2:
    w′(e) = w(e) − m(v).
    Then, T = (V, E′) is an RDMST of g = (V, E, w) rooted at r if and only if T is an RDMST of g = (V, E, w′) rooted
    at r.

    Input:
    graph: a weighted digraph graph in the reversed representation
    root: the root node
    Output: None
    """
    for node, edges in rgraph.items():
        if node == root:
            continue
        if edges:
            min_weight = min(edges.values())
        else:
            min_weight = 0
        for tail in edges.keys():
            edges[tail] -= min_weight


# g = {0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {0: 20, 2: 8}, 4: {2: 20, 3: 4},
#      5: {1: 16, 3: 8}}
# modify_edge_weights(g, 0)
# print(g)
# actual = expected = {0: {}, 1: {0: 16, 4: 0}, 2: {0: 2, 1: 0}, 3: {0: 12, 2: 0}, 4: {2: 16, 3: 0}, 5: {1: 8, 3: 0}}

# g = {0: {}, 1: {0: 2, 4: 2}, 2: {0: 2, 1: 2}, 3: {
#     0: 2, 2: 2}, 4: {2: 2, 3: 2}, 5: {1: 2, 3: 2}}
# modify_edge_weights(g, 1)
# print(g)
# actual = expected = {0: {}, 1: {}, 2: {0: 0, 1: 0}, 3: {0: 0, 2: 0}, 4: {2: 0, 3: 0}, 5: {1: 0, 3: 0}}


def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    """
    This function computes an RDST candidate based on Lemma 1 of weighted digraph graph
    (in the reversed representation).

    Lemma 1: Let E′ = {me(u) : u ∈ (V \ {r})}.
    Then, either T = (V, E′) is an RDMST of g rooted at r or T contains a cycle.

    Input:
    rgraph: a weighted digraph graph in the reversed representation
    root: the root node

    Output: rdst: the RDST candidate it computed as a weighted digraph in the reversed representation
    """
    rdst_candidate = {node: {} for node in rgraph.keys()}
    for node, edges in rgraph.items():
        if node == root:
            continue
        if edges:
            min_weight = min(edges.values())
        else:
            continue
        for tail, weight in edges.items():
            if weight == min_weight:
                rdst_candidate[node][tail] = weight
                break
    return rdst_candidate


# g = {0: {}, 1: {0: 20, 4: 4}, 2: {0: 4, 1: 2}, 3: {
#      0: 20, 2: 8}, 4: {2: 20, 3: 4}, 5: {1: 16, 3: 8}}
# print(compute_rdst_candidate(g, 0))
# actual = expected = {0: {}, 1: {4: 4}, 2: {1: 2}, 3: {2: 8}, 4: {3: 4}, 5: {3: 8}}

# g1 = {0: {}, 1: {0: 2, 4: 2}, 2: {0: 2, 1: 2}, 3: {
#     0: 2, 2: 2}, 4: {2: 2, 3: 2}, 5: {1: 2, 3: 2}}
# print(compute_rdst_candidate(g1, 0))
# actual = expected = {0: {}, 1: {0: 2}, 2: {0: 2}, 3: {0: 2}, 4: {2: 2}, 5: {1: 2}}


def compute_cycle(rdst_candidate: dict) -> tuple:
    """
    This function takes as input a RDST candidate of a weighted digraph (in the reversed
    representation) and computes a cycle in it.

    Input: rdst_candidate: a RDST candidate of a weighted digraph
    Output: a tuple containing the nodes of a cycle in the graph.
    """
    for node in rdst_candidate:
        cycle = [node]
        cur_node = node
        while len(rdst_candidate[cur_node]) > 0:
            tail_node = next(iter(rdst_candidate[cur_node]))
            cur_node = tail_node
            if cur_node in cycle:  # if cur_node is found --> end of cycle
                return tuple(cycle[cycle.index(cur_node):])
            cycle.append(cur_node)
    return tuple()


# print(compute_cycle({0: {}, 1: {4: 4}, 2: {
#     1: 2}, 3: {2: 8}, 4: {3: 4}, 5: {3: 8}}))
#actual = expected = (1, 4, 3, 2)

# print(compute_cycle({0: {}, 1: {0: 2, 4: 2}, 2: {0: 2, 1: 2}, 3: {
#     0: 2, 2: 2}, 4: {2: 2, 3: 2}, 5: {1: 2, 3: 2}}))
#actual = expected = ()


def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    """
    This function takes as input a weighted digraph graph in the standard representation 
    and a cycle and returns (contracted_graph, cstar) where contracted_graph is the 
    digraph (in the standard representation) that results from contracting cycle and cstar 
    is the number of the new node added to replace the cycle.

    Input:
    graph: a weighted digraph graph in the standard representation
    cycle: a cycle in graph represented by a tuple

    Ouput: a 2-tuple: first element is contracted_graph, the 
    digraph (in the standard representation) that results from contracting cycle, and second
    element is cstar, the number of the new node added to replace the cycle.
    """
    contracted_graph = deepcopy(graph)
    cstar = max(contracted_graph.keys()) + 1
    contracted_graph[cstar] = {}
    to_cycle = set()  # nodes that have edge to the cycle
    from_cycle = set()  # nodes that have edge from the cycle

    for node, edges in graph.items():
        if node not in cycle:
            for head in edges:
                if head in cycle:
                    to_cycle.add(node)
    for node in cycle:
        for head in graph[node]:
            if head not in cycle:
                from_cycle.add(head)
        if node in contracted_graph:   # remove node in the cycle from the graph
            contracted_graph.pop(node)

    for node in to_cycle:  # remove edge from node to a node in the cycle
        for head in graph[node]:
            if head in cycle:
                contracted_graph[node].pop(head)

    for node in from_cycle:  # find the weight from cstar to nodes outside the cycle
        min_weight = float('inf')
        for node_0 in cycle:
            if node in graph[node_0]:
                min_weight = min(graph[node_0][node], min_weight)
        contracted_graph[cstar][node] = min_weight

    for node in to_cycle:  # find the weight from a node outside the cycle to cstar
        min_weight = float('inf')
        for head in graph[node]:
            if head in cycle:
                min_weight = min(graph[node][head], min_weight)
        contracted_graph[node][cstar] = min_weight

    return (contracted_graph, cstar)


# g0 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
#       4: {1: 4}, 5: {}}
# g0_reversed = reverse_digraph_representation(g0)
# g0_rdst = compute_rdst_candidate(g0_reversed, 0)
# print(contract_cycle(g0, compute_cycle(g0_rdst)))
# actual = expected = ({0: {6: 4}, 5: {}, 6: {5: 8}}, 6)

# contracted_graph2 = contract_cycle(
#     {0: {2: 4, 1: 6}, 1: {2: 10}, 2: {3: 10}, 3: {1: 10}}, (1, 2, 3))
# print(contracted_graph2)
# expected = actual = ({0: {4: 4}, 4: {}}, 4)


def expand_graph(graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    """
    This function takes as returns a weighted digraph (in standard representation) 
    that results from expanding the cycle in rdst_candidate.

    Input: 
    - graph:  the weighted digraph (in standard representation) whose cycle was contracted;  
    - rdst_candidate: the RDST candidate rdst_candidate as a weighted digraph, in 
    standard representation, that was computed on the contracted 
    version of original_graph; 
    - cycle: the tuple of nodes on the cycle that was contracted; 
    - cstar: the number that labels the node that replaces the contracted cycle. 

    Output:
    expanded_graph: a weighted digraph (in standard representation) that results from 
    expanding the cycle in rdst_candidate
    """
    expanded_graph = deepcopy(rdst_candidate)

    # replace cstar by the nodes in the cycle
    expanded_graph.pop(cstar)
    for node, edges in expanded_graph.items():
        if cstar in edges:
            edges.pop(cstar)
    for node in cycle:
        expanded_graph[node] = {}

    # replace the edge (u, c*) in T' by the single edge (u, v*) in E that was the “origin” of edge (u, c*)
    vstar = None
    # find vstar and connect edge from outside the cycle to vstar
    for node in rdst_candidate:
        if cstar in rdst_candidate[node]:
            to_cycle_node = None
            to_cycle_weight = float('inf')
            for head in graph[node]:
                if head in cycle:
                    if graph[node][head] < to_cycle_weight:
                        to_cycle_node = head
                        to_cycle_weight = graph[node][head]
            vstar = to_cycle_node
            expanded_graph[node][vstar] = rdst_candidate[node][cstar]

    # replace every edge (c*, v) by the single edge in E that was the origin of edge (c*, v)
    for head in rdst_candidate[cstar]:
        from_cycle_node = None
        from_cycle_weight = float('inf')
        for node in cycle:
            if head in graph[node]:
                if graph[node][head] < from_cycle_weight:
                    from_cycle_node = node
                    from_cycle_weight = graph[node][head]
        expanded_graph[from_cycle_node][head] = rdst_candidate[cstar][head]

    # add every edge whose two endpoints are on the cycle C, except for the edge incoming into node v*
    for i in range(len(cycle)-1, 0, -1):
        if cycle[i-1] != vstar:
            expanded_graph[cycle[i]][cycle[i-1]] = graph[cycle[i]][cycle[i-1]]
    if cycle[-1] != vstar:
        expanded_graph[cycle[0]][cycle[-1]] = graph[cycle[0]][cycle[-1]]
    return expanded_graph


# print(expand_graph({0: {1: 0}, 1: {2: 0}, 2: {2: 0, 1: 0}},
#                    {0: {3: 0}, 3: {}}, [1, 2], 3))
# actual = expected = {0: {1: 0}, 1: {2: 0}, 2: {}}

# g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
#      4: {1: 4}, 5: {}}
# g_rev = reverse_digraph_representation(g)
# rdst_cand_grev = compute_rdst_candidate(g_rev, 0)
# cycle = compute_cycle(rdst_cand_grev)
# contracted_graph, cstar = contract_cycle(g, cycle)
# print(expand_graph(g, contracted_graph, cycle, cstar))
# actual = expected = {0: {2: 4}, 5: {}, 1: {}, 4: {1: 4}, 3: {5: 8, 4: 4}, 2: {3: 8}}


def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).

        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.

        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """

    if root not in graph:
        print("The root node does not exist")
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print("The root does not reach every other node in the graph")
            return

    rdmst = compute_rdmst_helper(graph, root)

    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)


def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """

    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)

        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        #cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)

        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(
            rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst

# Test digraphs
# Notice that the RDMST itself might not be unique, but its weight is


g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {
    3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# Results for compute_rdmst(g0, 0):
# ({0: {1: 2, 2: 2, 3: 2}, 1: {5: 2}, 2: {4: 2}, 3: {}, 4: {}, 5: {}}, 10)

g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {
    3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# Results for compute_rdmst(g1, 0):
# ({0: {2: 4}, 1: {}, 2: {3: 8}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, 28)

g2 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
# Results for compute_rdmst(g2, 0):
# ({0: {2: 4}, 1: {}, 2: {1: 2}}, 6)

g3 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0,
                                                                                     4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
# Results for compute_rdmst(g3, 1):
# ({1: {3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {}, 5: {}}, 11.1)

g4 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1, 8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0},
      6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}
# Results for compute_rdmst(g4, 1):
# ({1: {8: 6.1, 3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {9: 6.1, 10: 5.0}, 5: {}, 6: {7: 0.0}, 7: {}, 8: {}, 9: {}, 10: {6: 0.0}}, 28.3)

# print(compute_rdmst(g0, 0))
# print(compute_rdmst(g1, 0))
# print(compute_rdmst(g2, 0))
# print(compute_rdmst(g3, 1))
# print(compute_rdmst(g4, 1))
