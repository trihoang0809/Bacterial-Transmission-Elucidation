# Jay Fu
# jlf10
# COMP 182 Spring 2023 - Homework 6, Problem 2

# You can import any standard library, as well as Numpy and Matplotlib.
# You can use helper functions from provided.py, and autograder.py,
# but they have to be copied over here.
from typing import Tuple
from collections import *
from copy import *


def reverse_digraph_representation(graph: dict) -> dict:
    """
    Reverses the representation of a weighted digraph.
    Arguments:
    graph -- a weighted digraph in standard dictionary representation.
    Returns:
    A weighted digraph in standard dictionary representation.
    """
    rgraph = {}
    for node in graph:
        rgraph[node] = {}
    for node in graph:
        for neighbor in graph[node]:
            rgraph[neighbor][node] = graph[node][neighbor]
    return rgraph


g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8},
     4: {1: 4}, 5: {}}
g1 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1, 8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0},
      6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}

# print(reverse_digraph_representation(g))
# print(reverse_digraph_representation(g1))


def modify_edge_weights(rgraph: dict, root: int) -> None:
    """
    Modifies the edge weights of a weighted digraph according to the lemma below
    w'(e) = w(e) - m(v) such that m(v) is the minimum weight of an edge whose head is node v
    Arguments:
    rgraph -- a reversed weighted digraph in dictionary representation.
    root -- a node in rgraph.
    """
    for node in rgraph:
        if node == root:
            continue
        if len(rgraph[node]) > 0:
            m = min(rgraph[node].values())
        else:
            m = 0
        for neighbor in rgraph[node]:
            rgraph[node][neighbor] -= m

# rgraph = reverse_digraph_representation(g)
# modify_edge_weights(rgraph, 0)
# print(rgraph)
# rgraph1 = reverse_digraph_representation(g1)
# modify_edge_weights(rgraph1, 0)
# print(rgraph1)


def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    """
    Computes the RDST candidate of a weighted digraph rooted at node root based on lemma 1.
    Arguments:
    rgraph -- a reversed weighted digraph in dictionary representation.
    root -- a node in rgraph.
    Returns:
    An RDST candidate of rgraph rooted at root.
    """
    rdst_candidate = {}
    for node in rgraph:
        rdst_candidate[node] = {}
    for node in rgraph:
        if node == root:
            continue
        if len(rgraph[node]) > 0:
            m = min(rgraph[node].values())
        else:
            continue
        for neighbor in rgraph[node]:
            if rgraph[node][neighbor] == m:
                rdst_candidate[node][neighbor] = rgraph[node][neighbor]
                break
    return rdst_candidate
# rdst_candidate = compute_rdst_candidate(rgraph, 0)
# print(rdst_candidate)
# rdst_candidate1 = compute_rdst_candidate(rgraph1, 0)
# print(rdst_candidate1)


def compute_cycle(rdst_candidate: dict) -> tuple:
    """
    Compute a cycle in a given weighted directed graph in reversed representation.

    Arguments:
    graph (dict): A dictionary representing the weighted directed graph in reversed representation,

    Returns:
    tuple: A tuple containing the nodes of a cycle in the graph.
    """
    for node in rdst_candidate:
        path = [node]
        curr_node = node
        while len(rdst_candidate[curr_node]) > 0:
            next_node = next(iter(rdst_candidate[curr_node]))
            curr_node = next_node
            if curr_node in path:
                return tuple(path[path.index(curr_node):])
            path.append(curr_node)
    return tuple()

# cycle = compute_cycle(rdst_candidate)
# cycle1 = compute_cycle(rdst_candidate1)


def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    """
    Contracts a cycle in a given weighted directed graph in standard representation.
    Arguments:
    graph (dict): A dictionary representing the weighted directed graph in standard representation.
    cycle (tuple): A tuple containing the nodes of a cycle in graph.
    Returns:
    A tuple containing a dictionary representing the weighted directed graph in standard representation 
    and the number of the new node added to replace the cycle
    """
    from_cycle_adj = set()  # list of nodes with edge from cycle to node
    to_cycle_adj = set()  # list of nodes with edge from node to cycle
    for node in cycle:
        for neighbor in graph[node]:
            if neighbor not in cycle:
                from_cycle_adj.add(neighbor)
    for node in graph:
        for neighbor in graph[node]:
            if neighbor in cycle and node not in cycle:
                to_cycle_adj.add(node)
    new_graph = deepcopy(graph)
    cstar = max(new_graph.keys()) + 1
    for node in cycle:
        # remove node from new_graph
        if node in new_graph:
            new_graph.pop(node)
    for node in to_cycle_adj:
        # remove edge from node to cycle
        for neighbor in new_graph[node].copy():
            if neighbor in cycle:
                new_graph[node].pop(neighbor)
    new_graph[cstar] = {}
    for node in from_cycle_adj:
        minWeight = float('inf')
        for node2 in cycle:
            if node in graph[node2]:
                minWeight = min(minWeight, graph[node2][node])
        # add edge from cstar to node
        new_graph[cstar][node] = minWeight
    for node in to_cycle_adj:
        minWeight = float('inf')
        for nbr in graph[node]:
            if nbr in cycle:
                minWeight = min(minWeight, graph[node][nbr])
        # add edge from node to cstar
        new_graph[node][cstar] = minWeight
    return new_graph, cstar

# contracted_graph = contract_cycle(g, cycle)
# #print(contracted_graph)
# contracted_graph1 = contract_cycle(g1, cycle1)
# #print(contracted_graph1)
# contracted_graph2 = contract_cycle({0: {},1: {2: 10}, 2: {3: 10}, 3: {1: 10}}, [1, 2, 3])
# print(contracted_graph2)


def expand_graph(graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    """
    Expands a graph by replacing a cycle with a new node.
    Arguments:
    graph (dict): A dictionary representing the weighted directed graph (in standard representation) whose cycle was contracted.
    rdst_candidate (dict): A dictionary representing the RDST candidate computed on the contracted version of graph.
    cycle (tuple): A tuple containing the nodes of the cycle that was contracted in graph.
    cstar (int): The number that labels the node that replaces the contracted cycle.
    Returns:
    A dictionary representing the weighted directed graph in standard representation.
    """
    new_graph = deepcopy(rdst_candidate)

    # remove node cstar from new_graph
    new_graph.pop(cstar)
    # remove cstar from all adj nodes in new_graph
    for node in new_graph:
        if cstar in new_graph[node]:
            new_graph[node].pop(cstar)
    # add nodes in cycle to new_graph
    for node in cycle:
        new_graph[node] = {}

    Vstar = None
    # replace edge (u,C*) with (u,V*) where V* is the node in cycle that u is connected to
    for node in rdst_candidate:
        if cstar in rdst_candidate[node]:
            # find which node in cycle node is connected to
            ending_node = None
            min_weight = float('inf')
            for nbr1 in graph[node]:
                if nbr1 in cycle:
                    if graph[node][nbr1] < min_weight:
                        ending_node = nbr1
                        min_weight = graph[node][nbr1]
            Vstar = ending_node
            new_graph[node][ending_node] = rdst_candidate[node][cstar]

    # replace edge (C*,v) with (V*,v) where V* is the node in cycle that v is connected to
    for nbr in rdst_candidate[cstar]:
        # find which node in cycle nbr is connected to
        starting_node = None
        min_weight = float('inf')
        for node in cycle:
            if nbr in graph[node]:
                if graph[node][nbr] < min_weight:
                    starting_node = node
                    min_weight = graph[node][nbr]
        new_graph[starting_node][nbr] = rdst_candidate[cstar][nbr]

    # add edges from nodes in cycle to nodes in rdst_candidate
    for i in range(len(cycle)-1, 0, -1):
        if cycle[i-1] != Vstar:
            new_graph[cycle[i]][cycle[i-1]] = graph[cycle[i]][cycle[i-1]]
    if cycle[-1] != Vstar:
        new_graph[cycle[0]][cycle[-1]] = graph[cycle[0]][cycle[-1]]
    return new_graph

# print(expand_graph({0: {1: 0}, 1: {2: 0}, 2: {2: 0, 1: 0}}, {0: {3: 0}, 3: {}}, [1, 2], 3))
# cycle = compute_cycle(g)
# contracted_graph,cstar = contract_cycle(g, cycle)
# print(expand_graph(g, contracted_graph, cycle, cstar))


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
        # print(reverse_digraph_representation(rgraph))
        # print(new_rdst_candidate)
        # print(cycle)
        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(
            rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst


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

# print(compute_genetic_distance("00000000000000000000000000000000000001011","00000000000000000000000001000000011110000"))
# print(compute_genetic_distance("00000000000000000000000000000000011110000","00010110000001001101000010001000100001011"))


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

#print(construct_complete_weighted_digraph("patient_sequences.txt", "patient_traces.txt"))


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


print(infer_transmap("patient_sequences.txt", "patient_traces.txt", 1))
