from statistics import mean, median, stdev

from pm4py.visualization.common.utils import *


def get_shortest_paths_from_trans(original_trans, trans, spaths, visited_arcs, visited_transitions, added_elements,
                                  rec_depth):
    """
    Get shortest paths from a given transition

    Parameters
    --------------
    original_trans
        Original transition
    trans
        Current considered transition
    spaths
        Map of shortest paths
    visited_arcs
        Set of visited arcs
    visited_transitions
        Set of visited transitions
    added_elements
        Elements to add recursively
    rec_depth
        Recursion depth

    Returns
    -------------
    spaths
        Map of shortest paths
    visited_arcs
        Set of visited arcs
    added_elements
        Elements to add recursively
    """
    for out_arc in trans.out_arcs:
        if out_arc not in visited_arcs:
            visited_arcs.add(out_arc)
            target_place = out_arc.target
            for place_out_arc in target_place.out_arcs:
                if place_out_arc not in visited_arcs:
                    visited_arcs.add(place_out_arc)
                    target_trans = place_out_arc.target
                    if target_trans not in visited_transitions:
                        visited_transitions.add(target_trans)
                        if target_trans.label:
                            el1 = ((original_trans.name, target_trans.name), 0, rec_depth)
                            if out_arc not in spaths:
                                spaths[out_arc] = set()
                            spaths[out_arc].add(el1)
                            added_elements.add(el1)
                            el2 = ((original_trans.name, target_trans.name), 1, rec_depth)
                            if place_out_arc not in spaths:
                                spaths[place_out_arc] = set()
                            spaths[place_out_arc].add(el2)
                            added_elements.add(el2)
                        else:
                            spaths, visited_arcs, visited_transitions, added_elements = get_shortest_paths_from_trans(
                                original_trans,
                                target_trans, spaths,
                                visited_arcs,
                                visited_transitions,
                                added_elements,
                                rec_depth + 1)
                            for element in added_elements:
                                new_element = list(element)
                                if new_element[1] == 0:
                                    new_element[1] = 2
                                    if out_arc not in spaths:
                                        spaths[out_arc] = set()
                                    spaths[out_arc].add(tuple(new_element))
                                if new_element[1] == 1:
                                    new_element[1] = 3
                                    if place_out_arc not in spaths:
                                        spaths[place_out_arc] = set()
                                    spaths[place_out_arc].add(tuple(new_element))
    return spaths, visited_arcs, visited_transitions, added_elements


def get_shortest_paths(net, enable_extension=False):
    """
    Gets shortest paths between visible transitions in a Petri net

    Parameters
    -----------
    net
        Petri net
    enable_extension
        Enable decoration of more arcs, in a risky way, when needed

    Returns
    -----------
    spaths
        Shortest paths
    """
    spaths = {}
    for trans in net.transitions:
        if trans.label:
            visited_arcs = set()
            visited_transitions = set()
            added_elements = set()
            spaths, visited_arcs, visited_transitions, added_elements = get_shortest_paths_from_trans(trans, trans,
                                                                                                      spaths,
                                                                                                      visited_arcs,
                                                                                                      visited_transitions,
                                                                                                      added_elements, 0)
    spaths_keys = list(spaths.keys())
    for edge in spaths_keys:
        list_zeroones = [el for el in spaths[edge] if el[1] == 0 or el[1] == 1]
        if list_zeroones:
            spaths[edge] = {x for x in spaths[edge] if x[0] == 0 or x[1] == 1}
        else:
            unique_targets = set([x[0] for x in spaths[edge]])
            if len(unique_targets) == 1:
                spaths[edge] = set()
                spaths[edge].add((list(unique_targets)[0], 0, 0))
            else:
                if enable_extension:
                    min_dist = min([x[2] for x in spaths[edge]])
                    targets = set([x[0] for x in spaths[edge] if x[2] == min_dist])
                    spaths[edge] = set()
                    for target in targets:
                        spaths[edge].add((target, 0, min_dist))
                else:
                    del spaths[edge]

    return spaths


def get_decorations_from_dfg_spaths_acticount(net, dfg, spaths, activities_count, variant="frequency",
                                              aggregation_measure=None):
    """
    Get decorations from Petrinet without doing any replay
    but based on DFG measures, shortest paths and activities count.
    The variant could be 'frequency' or 'performance'.
    Aggregation measure could also be specified

    Parameters
    -----------
    net
        Petri net
    dfg
        Directly-Follows graph
    spaths
        Shortest paths between visible transitions in the Petri net
    activities_count
        Count of activities in the Petri net
    variant
        Describe how to decorate the Petri net (could be frequency or performance)
    aggregation_measure
        Specifies the aggregation measure

    Returns
    -----------
    decorations
        Decorations to use for the Petri net
    """
    decorations_single_contrib = {}
    decorations_single_contrib_trans = {}
    decorations_int = {}
    decorations = {}
    if aggregation_measure is None:
        if "frequency" in variant:
            aggregation_measure = "sum"
        elif "performance" in variant:
            aggregation_measure = "mean"
    for arc in spaths:
        for couple in spaths[arc]:
            dfg_key = couple[0]
            if dfg_key in dfg:
                if arc not in decorations_single_contrib:
                    decorations_single_contrib[arc] = []
                decorations_single_contrib[arc].append(dfg[dfg_key])
                if dfg_key[1] not in decorations_single_contrib_trans:
                    decorations_single_contrib_trans[dfg_key[1]] = {}
                decorations_single_contrib_trans[dfg_key[1]][dfg_key[0]] = dfg[dfg_key]
    for arc in decorations_single_contrib:
        decorations_value = None
        if aggregation_measure == "sum":
            decorations_value = sum(decorations_single_contrib[arc])
        elif aggregation_measure == "mean":
            decorations_value = mean(decorations_single_contrib[arc])
        elif aggregation_measure == "median":
            decorations_value = median(decorations_single_contrib[arc])
        elif aggregation_measure == "stdev":
            decorations_value = stdev(decorations_single_contrib[arc])
        elif aggregation_measure == "min":
            decorations_value = min(decorations_single_contrib[arc])
        elif aggregation_measure == "max":
            decorations_value = max(decorations_single_contrib[arc])
        if decorations_value is not None:
            decorations_int[arc] = decorations_value

    if decorations_int:
        arcs_min_value = min(list(decorations_int.values()))
        arcs_max_value = max(list(decorations_int.values()))
        for arc in decorations_int:
            if "performance" in variant:
                arc_label = human_readable_stat(decorations_int[arc])
            else:
                arc_label = str(decorations_int[arc])
            decorations[arc] = {"label": arc_label,
                                "penwidth": str(get_arc_penwidth(decorations_int[arc], arcs_min_value, arcs_max_value))}
        trans_map = {}
        for trans in net.transitions:
            if trans.label:
                trans_map[trans.label] = trans
        if "frequency" in variant:
            act_min_value = min(list(activities_count.values()))
            act_max_value = max(list(activities_count.values()))
            for act in activities_count:
                if act in trans_map:
                    trans = trans_map[act]
                    color = get_trans_freq_color(activities_count[act], act_min_value, act_max_value)
                    label = act + " (" + str(activities_count[act]) + ")"
                    decorations[trans] = {"label": label, "color": color}
        elif "performance" in variant:
            for act in decorations_single_contrib_trans:
                if act in trans_map:
                    trans = trans_map[act]
                    trans_values = list(decorations_single_contrib_trans[act].values())
                    decorations[trans] = {"performance": mean(trans_values)}

    return decorations
