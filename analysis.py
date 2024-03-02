# Tri Hoang
# th52
# COMP 182 Spring 2023 - Homework 5, Problem 4

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.

# Your code here...
from typing import Tuple
from collections import *
from copy import *

# PROVIDED/WRITTEN FUNCTIONS


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


def infer_transmap(gen_data, epi_data, patient_id):
    """
        Infers a transmission map based on genetic
        and epidemiological data rooted at patient_id

        Arguments:
        gen_data -- filename with genetic data for each patient
        epi_data -- filename with epidemiological data for each patient
        patient_id -- the id of the 'patient 0'

        Returns:
        The most likely transmission map for the given scenario as the RDMST 
        of a weighted, directed, complete digraph
        """

    complete_digraph = construct_complete_weighted_digraph(gen_data, epi_data)
    return compute_rdmst(complete_digraph, patient_id)


def read_patient_sequences(filename):
    """
        Turns the bacterial DNA sequences (obtained from patients) into a list containing tuples of
        (patient ID, sequence).

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A list of (patient ID, sequence) tuples.
        """
    sequences = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            if len(line) > 5:
                patient_num, sequence = line.split("\t")
                sequences.append((int(patient_num), ''.join(
                    e for e in sequence if e.isalnum())))
    return sequences


def read_patient_traces(filename):
    """
        Reads the epidemiological data file and computes the pairwise epidemiological distances between patients

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A dictionary of dictionaries where dict[i][j] is the
        epidemiological distance between i and j.
    """
    trace_data = []
    patient_ids = []
    first_line = True
    with open(filename) as f:
        for line in f:
            if first_line:
                patient_ids = line.split()
                patient_ids = list(map(int, patient_ids))
                first_line = False
            elif len(line) > 5:
                trace_data.append(line.rstrip('\n'))
    return compute_pairwise_epi_distances(trace_data, patient_ids)


def compute_pairwise_gen_distances(sequences, distance_function):
    """
        Computes the pairwise genetic distances between patients (patients' isolate genomes)

        Arguments:
        sequences -- a list of sequences that correspond with patient id's
        distance_function -- the distance function to apply to compute the weight of the 
        edges in the returned graph

        Returns:
        A dictionary of dictionaries where gdist[i][j] is the
        genetic distance between i and j.
        """
    gdist = {}
    cultures = {}

    # Count the number of differences of each sequence
    for i in range(len(sequences)):
        patient_id = sequences[i][0]
        seq = sequences[i][1]
        if patient_id in cultures:
            cultures[patient_id].append(seq)
        else:
            cultures[patient_id] = [seq]
            gdist[patient_id] = {}
    # Add the minimum sequence score to the graph
    for pat1 in range(1, max(cultures.keys()) + 1):
        for pat2 in range(pat1 + 1, max(cultures.keys()) + 1):
            min_score = float("inf")
            for seq1 in cultures[pat1]:
                for seq2 in cultures[pat2]:
                    score = distance_function(seq1, seq2)
                    if score < min_score:
                        min_score = score
            gdist[pat1][pat2] = min_score
            gdist[pat2][pat1] = min_score
    return gdist


### HELPER FUNCTIONS. ###

def find_first_positives(trace_data):
    """
        Finds the first positive test date of each patient
        in the trace data.
        Arguments:
        trace_data -- a list of data pertaining to location
        and first positive test date
        Returns:
        A dictionary with patient id's as keys and first positive
        test date as values. The date numbering starts from 0 and
        the patient numbering starts from 1.
        """
    first_pos = {}
    for pat in range(len(trace_data[0])):
        first_pos[pat + 1] = None
        for date in range(len(trace_data)):
            if trace_data[date][pat].endswith(".5"):
                first_pos[pat + 1] = date
                break
    return first_pos


def compute_epi_distance(pid1, pid2, trace_data, first_pos1, first_pos2, patient_ids):
    """
        Computes the epidemiological distance between two patients.

        Arguments:
        pid1 -- the assumed donor's index in trace data
        pid2 -- the assumed recipient's index in trace data
        trace_data -- data for days of overlap and first positive cultures
        first_pos1 -- the first positive test day for pid1
        first_pos2 -- the first positive test day for pid2
        patient_ids -- an ordered list of the patient IDs given in the text file

        Returns:
        Finds the epidemiological distance from patient 1 to
        patient 2.
        """
    first_overlap = -1
    assumed_trans_date = -1
    pid1 = patient_ids.index(pid1)
    pid2 = patient_ids.index(pid2)
    # Find the first overlap of the two patients
    for day in range(len(trace_data)):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            first_overlap = day
            break
    if (first_pos2 < first_overlap) | (first_overlap < 0):
        return len(trace_data) * 2 + 1
    # Find the assumed transmission date from patient 1 to patient 2
    for day in range(first_pos2, -1, -1):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            assumed_trans_date = day
            break
    sc_recip = first_pos2 - assumed_trans_date

    if first_pos1 < assumed_trans_date:
        sc_donor = 0
    else:
        sc_donor = first_pos1 - assumed_trans_date
    return sc_donor + sc_recip


def compute_pairwise_epi_distances(trace_data, patient_ids):
    """
        Turns the patient trace data into a dictionary of pairwise 
        epidemiological distances.

        Arguments:
        trace_data -- a list of strings with patient trace data
        patient_ids -- ordered list of patient IDs to expect

        Returns:
        A dictionary of dictionaries where edist[i][j] is the
        epidemiological distance between i and j.
        """
    edist = {}
    proc_data = []
    # Reformat the trace data
    for i in range(len(trace_data)):
        temp = trace_data[i].split()[::-1]
        proc_data.append(temp)
    # Find first positive test days and remove the indication from the data
    first_pos = find_first_positives(proc_data)
    for pid in first_pos:
        day = first_pos[pid]
        proc_data[day][pid - 1] = proc_data[day][pid - 1].replace(".5", "")
    # Find the epidemiological distance between the two patients and add it
    # to the graph
    for pid1 in patient_ids:
        edist[pid1] = {}
        for pid2 in patient_ids:
            if pid1 != pid2:
                epi_dist = compute_epi_distance(pid1, pid2, proc_data,
                                                first_pos[pid1], first_pos[pid2], patient_ids)
                edist[pid1][pid2] = epi_dist
    return edist


# NEW FUNCTIONS

def compute_genetic_distance(seq1, seq2):
    """
    This function takes as arguments two sequences
    (from the list built by read_patient_sequences) and returns their Hamming distance.
    """
    hamming_distance = 0
    i = 0
    while i < len(seq1):
        if seq1[i] != seq2[i]:
            hamming_distance += 1
        i += 1
    return hamming_distance


# print(compute_genetic_distance("00101", "10100")) --> 2
# print(compute_genetic_distance("00101", "00101")) --> 0
# print(compute_genetic_distance("00000000000000000000000000000000000001011",
#       "00000000000000000000000001000000011110000")) --> 8

def construct_complete_weighted_digraph(gen_file: str, epi_file: str) -> dict:
    """
    This function takes as arguments the filenames of the genetic data and 
    epidemiological data (in this order) and returns a complete, weighted, 
    digraph whose nodes are the patients (use the patient id's for node labels) 
    and whose edge weights are based on Equation (1) of the description.

    Input:
    - gen_file: the filename of the genetic data
    - epi_file: the filename of the epidemiological data

    Output:
    graph: a complete, weighted, digraph represented by a dictionary
    """
    graph = {}
    gen_data = read_patient_sequences(gen_file)
    gen_distances = compute_pairwise_gen_distances(
        gen_data, compute_genetic_distance)
    epi_data = read_patient_traces(epi_file)

    # add patients as nodes to graph
    for patient in gen_data:
        graph[patient[0]] = {}

    # calculate max_E in the equation
    max_e = -1
    for patient, others in epi_data.items():
        for other_patient in others:
            if others[other_patient] > max_e:
                max_e = others[other_patient]

    for patient_a in graph:
        for patient_b in graph:
            if patient_a != patient_b:
                # G_AB in the equation
                g_ab = gen_distances[patient_a][patient_b]
                # the equation inside the parentheses
                e_ab = epi_data[patient_a][patient_b]
                # D_AB, or weight of (A, B)
                d_ab = g_ab + ((999 * (e_ab / max_e)) / 100000)
                graph[patient_a][patient_b] = d_ab
    return graph


print(construct_complete_weighted_digraph(
    "patient_sequences.txt", "patient_traces.txt"))
#print(infer_transmap("patient_sequences.txt", "patient_traces.txt", 1))
# -->({1: {8: 6.00999, 4: 9.00999, 3: 7.001112687651331}, 8: {}, 9: {},
#      4: {9: 6.00999, 10: 5.000822421307506}, 11: {}, 18: {}, 16: {}, 13: {18: 2.00999},
#      12: {13: 0.00999}, 15: {16: 2.0}, 14: {15: 0.0024914527845036317}, 17: {14: 0.00123363196125908},
#      10: {11: 3.0007498547215494, 12: 2.0, 6: 0.0005079661016949153}, 6: {7: 0.0},
#      7: {17: 0.0020076755447941885}, 2: {}, 5: {2: 1.0005321549636803}, 3: {5: 0.0}}, 43.05940784503632)
