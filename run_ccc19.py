from copy import deepcopy

from pm4py.algo.conformance.tokenreplay import factory as tr_rep_factory
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.importer.xes import factory as xes_importer
from pm4py.objects.petri.importer import pnml as petri_importer
from scipy.stats import pearsonr
from pm4py.algo.enhancement.sna import factory as sna_factory
from pm4py.visualization.sna import factory as sna_vis_factory
from pm4py.algo.discovery.dfg import factory as dfg_factory
from pm4py.algo.simulation.playout import factory as playout_factory
from pm4py.visualization.dfg import factory as dfg_vis_factory
from pm4py.visualization.petrinet import factory as pn_vis_factory
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.conversion.log import factory as log_conv_factory
from pm4py.objects.random_variables.random_variable import RandomVariable
from pm4py.util import constants
from pm4py.objects.log.log import EventLog, Trace


def print_dictio(round_trans_fit, resources, activities):
    tabular = "\\begin{tabular}{l|"
    header = "{\\bf Resource} & "
    for index, act in enumerate(activities):
        if index > 0:
            header = header + " & "
        tabular = tabular + "c"
        header = header + "\\rot[90]{" + act + "}"
    tabular = tabular + "|c}\n\\hline"
    header = header + " & \\rot[90]{\\bf{Sum}} \\\\"
    print(tabular)
    print(header)
    print("\\hline")
    for res in resources:
        line = res.replace("_", "\\_") + " & "
        sum_res = 0
        for index, act in enumerate(activities):
            if index > 0:
                line = line + " & "
            line = line + str(round_trans_fit[res][act])
            sum_res = sum_res + round_trans_fit[res][act]
        line = line + " & " + str(sum_res) + " \\\\"
        print(line)
    bottom = "\\hline\n{\\bf Sum}"
    for act in activities:
        sum_act = 0
        for res in resources:
            sum_act = sum_act + round_trans_fit[res][act]
        bottom = bottom + " & " + str(sum_act)
    bottom = bottom + " & ~ \\\\ \n\\hline"
    print(bottom)


def calculate_diff(pre, post, resources, activities):
    diff = {}
    for res in resources:
        diff[res] = {}
        for act in activities:
            diff[res][act] = post[res][act] - pre[res][act]
    return diff


def calculate_pearson_features(activities, resource, res_act):
    fea = [0] * len(activities)

    for index, act in enumerate(activities):
        fea[index] = res_act[resource][act]

    return fea


def calculate_pearson(dictio, resources, activities):
    for r1 in resources:
        pears_feat1 = calculate_pearson_features(activities, r1, dictio)
        r1_values_list = []
        for r2 in resources:
            if not r1 == r2:
                pears_feat2 = calculate_pearson_features(activities, r2, dictio)
                r, s = pearsonr(pears_feat1, pears_feat2)
                r1_values_list.append([r2,r])
        r1_values_list = sorted(r1_values_list, key=lambda x: x[1], reverse=True)
        r1_values_list = [x for x in r1_values_list if x[1] > 0.75]
        #print(r1,r1_values_list)

def calculate_dfg_diff(dfg1, dfg2):
    dfg = deepcopy(dfg1)
    for key in dfg2:
        if key in dfg.keys():
            del dfg[key]
    return dfg



parameters_filt_transition = {constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "lifecycle:transition"}
parameters_filt_round = {constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "ROUND"}
parameters_filt_resource = {constants.PARAMETER_CONSTANT_ATTRIBUTE_KEY: "org:resource"}

log = xes_importer.apply("ccc19.xes")
log = attributes_filter.apply_events(log, ["complete"], parameters=parameters_filt_transition)
av = attributes_filter.get_attribute_values(log, "concept:name")
all_values = {}
act_rv = {}
lv_min = {}
lv_max = {}
for activity in av.keys():
    flog = attributes_filter.apply_events(log, [activity])
    values = []
    for trace in flog:
        i = 0
        while i < len(trace):
            values.append(int(trace[i]["VIDEOEND"]) - int(trace[i]["VIDEOSTART"]))
            i = i + 1
    values = sorted(values)
    v = RandomVariable()
    v.calculate_parameters(values)
    act_rv[activity] = v
    all_values[activity] = values
    lv_min[activity] = (v.calculate_loglikelihood([values[0], values[0]]))/2.0
    lv_max[activity] = (v.calculate_loglikelihood([values[-1], values[-1]])) / 2.0
#print(act_rv)
#print(median_values)
#print(all_values)
#print(act_rv)
#print(lv_min)
#print(lv_max)
#print(median_llh)
#print(min_llh)
new_log = EventLog()
i = 0
while i < len(log):
    j = 0
    while j < len(log[i]):
        trace = Trace()
        activity = log[i][j]["concept:name"]
        diff = int(log[i][j]["VIDEOEND"]) - int(log[i][j]["VIDEOSTART"])
        rv = act_rv[activity]
        if not rv.get_transition_type() == "IMMEDIATE":
            likelihood = rv.calculate_loglikelihood([diff, diff])
            if diff > 0 and likelihood < -8:
                pass
            else:
                trace.append(log[i][j])
        else:
            trace.append(log[i][j])
        new_log.append(trace)
        j = j + 1
    i = i + 1
log = new_log
net, im, fm = petri_importer.import_net("ccc19.pnml")
gviz = pn_vis_factory.apply(net, im, fm)
pn_vis_factory.save(gviz, "ccc19.png")
pre_round_log = attributes_filter.apply(deepcopy(log), ["Pre"], parameters=parameters_filt_round)
post_round_log = attributes_filter.apply(deepcopy(log), ["Post"], parameters=parameters_filt_round)
pre_round_resource_activity = {}
post_round_resource_activity = {}
resources = sorted(attributes_filter.get_attribute_values(log, "org:resource"))
activities = sorted(attributes_filter.get_attribute_values(log, "concept:name"))
for res in resources:
    pre_round_resource_activity[res] = {}
    post_round_resource_activity[res] = {}
    for trans in activities:
        pre_round_resource_activity[res][trans] = 0
        post_round_resource_activity[res][trans] = 0

aligned_traces_pre, place_fitness_per_trace_pre, transition_fitness_per_trace_pre, notexisting_activities_in_model_pre = tr_rep_factory.apply(
    pre_round_log, net, im, fm,
    parameters={"enable_pltr_fitness": True, "disable_variants": True, "cleaning_token_flood": True})
aligned_traces_post, place_fitness_per_trace_post, transition_fitness_per_trace_post, notexisting_activities_in_model_post = tr_rep_factory.apply(
    post_round_log, net, im, fm,
    parameters={"enable_pltr_fitness": True, "disable_variants": True, "cleaning_token_flood": True})

for trans in transition_fitness_per_trace_pre:
    for trace in transition_fitness_per_trace_pre[trans]["underfed_traces"]:
        for ex in transition_fitness_per_trace_pre[trans]["underfed_traces"][trace]:
            pre_round_resource_activity[ex["org:resource"]][trans.label] = \
                pre_round_resource_activity[ex["org:resource"]][trans.label] + 1

for trans in transition_fitness_per_trace_post:
    for trace in transition_fitness_per_trace_post[trans]["underfed_traces"]:
        for ex in transition_fitness_per_trace_post[trans]["underfed_traces"][trace]:
            post_round_resource_activity[ex["org:resource"]][trans.label] = \
                post_round_resource_activity[ex["org:resource"]][trans.label] + 1

diff = calculate_diff(pre_round_resource_activity, post_round_resource_activity, resources, activities)
#calculate_pearson(pre_round_resource_activity, resources, activities)
#print_dictio(diff, resources, activities)

#hw_metric_pre = sna_factory.apply(pre_round_log, variant="handover")
#hw_metric_post = sna_factory.apply(post_round_log, variant="handover")
#sub_metric_pre = sna_factory.apply(pre_round_log, variant="subcontracting")
#sub_metric_post = sna_factory.apply(post_round_log, variant="subcontracting")

#gviz = sna_vis_factory.apply(hw_metric_post, variant="pyvis")
#sna_vis_factory.view(gviz, variant="pyvis")

# for act in activities:
#    for res in resources:
#

# print(pre_round_resource_activity)
# print(post_round_resource_activity)

admitted_behavior_model = dfg_factory.apply(playout_factory.apply(net, im, fm))
dfg_behavior_pre = dfg_factory.apply(pre_round_log)
dfg_behavior_post = dfg_factory.apply(post_round_log)

#dfg_behavior_pre_diff_admitted = calculate_dfg_diff(dfg_behavior_pre, admitted_behavior_model)
#keys = list(dfg_behavior_pre_diff_admitted.keys())
#for key in keys:
#    if dfg_behavior_pre_diff_admitted[key] < 3:
#        del dfg_behavior_pre_diff_admitted[key]
#gviz_dfg_pre_diff = dfg_vis_factory.apply(dfg_behavior_pre_diff_admitted, log=pre_round_log, parameters={"format": "svg"})
#dfg_vis_factory.view(gviz_dfg_pre_diff)

#dfg_behavior_post_diff_admitted = calculate_dfg_diff(dfg_behavior_post, admitted_behavior_model)
#keys = list(dfg_behavior_post_diff_admitted.keys())
#for key in keys:
#    if dfg_behavior_post_diff_admitted[key] < 3:
#        del dfg_behavior_post_diff_admitted[key]
#gviz_dfg_post_diff = dfg_vis_factory.apply(dfg_behavior_post_diff_admitted, log=pre_round_log, parameters={"format": "svg"})
#dfg_vis_factory.view(gviz_dfg_post_diff)

#dfg_behavior_pre_diff_admitted = calculate_dfg_diff(dfg_behavior_pre, admitted_behavior_model)
#dfg_behavior_pre_diff_admitted = calculate_dfg_diff(dfg_behavior_pre_diff_admitted, dfg_behavior_post)
#keys = list(dfg_behavior_pre_diff_admitted.keys())
#for key in keys:
#    if dfg_behavior_pre_diff_admitted[key] < 3:
#        del dfg_behavior_pre_diff_admitted[key]
#gviz_dfg_pre_diff = dfg_vis_factory.apply(dfg_behavior_pre_diff_admitted, log=pre_round_log, parameters={"format": "svg"})
#dfg_vis_factory.view(gviz_dfg_pre_diff)

#for resource in resources:
#    resource_pre_log = attributes_filter.apply(pre_round_log, [resource], parameters=parameters_filt_resource)
#    resource_post_log = attributes_filter.apply(pre_round_log, [resource], parameters=parameters_filt_resource)

#    dfg_behavior_pre = dfg_factory.apply(resource_pre_log)
#    dfg_behavior_post = dfg_factory.apply(resource_post_log)

#    dfg_behavior_pre_diff_admitted = calculate_dfg_diff(dfg_behavior_pre, admitted_behavior_model)

#    print(resource)
#    gviz_dfg_pre_diff = dfg_vis_factory.apply(dfg_behavior_pre_diff_admitted, log=pre_round_log,
#                                              parameters={"format": "svg"})

#    dfg_vis_factory.view(gviz_dfg_pre_diff)

#    input()
