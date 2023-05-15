from Entity import Entity
from Event import Event
from Relation import Whole_Member_Relation, Set_Member_Relation, Bridging

import xml.etree.ElementTree as ET
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict



def find_instances(filename):
    # TODO: Add missing instances
    wanted_instances = ["ENTITY", "EVENT"]
    wanted_relations = ["SET/MEMBER", "WHOLE/PART", "BRIDGING"]
    tree = ET.parse(filename)
    root = tree.getroot()
    #print(root)
    entities = []
    #events = []
    relations = []
    # retrieve the annotations
    for node in root.find("annotations"):
        for ele in node:
            if ele.tag == "type":
                if ele.text == "ENTITY": # get entities from xml
                    ent = extract_enities(node, ele.text)
                    if ent is not None:
                        entities.append(ent)
                if ele.text == "EVENT": # get events from xml
                    event = extract_event(node, ele.text)
                    if event is not None:
                        entities.append(event)
                if ele.text in wanted_relations: # retrieve the relations
                    relation = extract_relations(node, ele.text, entities, event)
                    relations.append(relation)
    return relations, entities



def extract_relations(node, type, entities, event):
    id = ""
    for ele in node:
        if ele.tag == "id":
            id = ele.text
        if type == "BRIDGING" and ele.tag == "properties":
            arg = ""
            rel_to = ""
            for i in range(len(ele)):
                if ele[i].tag == "Argument":
                    arg = ele[i].text
                if ele[i].tag == "Related_to":
                    rel_to = ele[i].text
            bridge = Bridging(id, type, arg, rel_to)
            return bridge
        if type == "SET/MEMBER" and ele.tag == "properties":
            se = ""
            mem1= ""
            mem2 = ""
            for i in range(len(ele)):
                if ele[i].tag == "Set":
                    se = ele[i].text
                if ele[i].tag == "Member" and mem1 == "":
                    mem1 = ele[i].text
                if ele[i].tag == "Member" and mem2 == "" and i != 1:
                    mem2 = ele[i].text
            set_mem = Set_Member_Relation(id, type, se, mem1, mem2)
            return set_mem
        if type == "WHOLE/PART" and ele.tag == "properties":
            whole = ""
            part = ""
            for i in range(len(ele)):
                if ele[i].tag == "Whole":
                    whole = ele[i].text
                if ele[i].tag == "Part":
                    part = ele[i].text
            whole_mem = Whole_Member_Relation(id, type, whole, part)
            return whole_mem



def extract_enities(node, type):
    """
    Function takes in a node and extracts important informations like span or type. Creates an enitity object.
    :param node:
    :return:
    """
    id = ""
    start = 0
    end = 0
    # retrieve the necessary informations
    # TODO: check for POS
    for ele in node:
        if ele.tag == "id":
            id = ele.text
        if ele.tag == "span":
            start = int(ele.text.split(",")[0])
            end = int(ele.text.split(",")[1])
        if ele.tag == "properties":
            for su in ele:
                if su.tag == "Polarity":
                    if su.text != "POS": # filter for != POS
                        return None
                if su.tag == "ContextualModality":
                    if su.text == "HYPOTHETICAL":
                        return None
    # create object
    entity = Entity(id, start, end, type)
    return entity


def extract_event(node, type):
    id = ""
    start = 0
    end = 0
    for ele in node:
        if ele.tag == "id":
            id = ele.text
        if ele.tag == "span":
            start = int(ele.text.split(",")[0])
            end = int(ele.text.split(",")[1])
    # create object
    event = Event(id, start, end, type)
    return event


def extract_from_source(rel, ent, data):
    with open(data, "r", encoding="utf-8") as f:
        data = f.read()
    data = data.lower() # NEU
    rel = rel[::-1]
    ent = ent[::-1]
    print(len(ent))
    id_to_words = defaultdict(str)
    id_to_span = defaultdict(list)
    words = data.split()
    print(data)
    for i in range(len(ent)): # retrieve the words, which match the span
        #print(data[ent[i].start:ent[i].end])
        id_to_words[ent[i].id] = data[ent[i].start:ent[i].end]
        id_to_span[ent[i].id] = [ent[i].start, ent[i].end]
    # über startwerte verlgeichen, dann ändern
    filtered_inst, entity_to_role = sort_and_filter_instances(rel, ent, id_to_words, id_to_span)
    #"""
    for inst in filtered_inst:
        print(inst, id_to_span[inst], entity_to_role[inst])
        print(data[id_to_span[inst][0]:id_to_span[inst][1]])
        data = data[:id_to_span[inst][0]] + entity_to_role[inst] + data[id_to_span[inst][1]:]
        #data[id_to_span[inst][0]:id_to_span[inst][1]] = entity_to_role[inst]

        #print(data)
    print(data)
    return data


def sort_and_filter_instances(rel, ent, id_to_words, id_to_span):
    """
    When manipulating the string, the spans will change. Therefore, the instances have to be sorted and the string
    will be processed reversed.
    """
    # filter
    entities_used_by_relations = []
    entitiy_to_role = {}
    for i in range(len(rel)):
        if rel[i].typ == "WHOLE/PART":
            entities_used_by_relations.append(rel[i].ent1)
            entitiy_to_role[rel[i].ent1] = "Whole"
            entities_used_by_relations.append(rel[i].ent2)
            entitiy_to_role[rel[i].ent2] = "Part"
        if rel[i].typ == "SET/MEMBER":
            entities_used_by_relations.append(rel[i].set)
            entitiy_to_role[rel[i].set] = "Set"
            entities_used_by_relations.append(rel[i].mem1)
            entitiy_to_role[rel[i].mem1] = "Member"
            entities_used_by_relations.append(rel[i].mem2)
            entitiy_to_role[rel[i].mem2] = "Member"
        if rel[i].typ == "BRIDGING":
            entities_used_by_relations.append(rel[i].arg)
            entitiy_to_role[rel[i].arg] = "Argument"
            entities_used_by_relations.append(rel[i].related_to)
            entitiy_to_role[rel[i].related_to] = "Related_to"

    #print("ent used by rel: ", len(entities_used_by_relations))
    #print(entities_used_by_relations)
    #print(id_to_span)
    filtered = {}
    for key,val in id_to_span.items():
        if key in entities_used_by_relations:
            filtered[key] = val
    filtered = {k: v for k, v in sorted(filtered.items(), key=lambda item: item[1][0], reverse = True)}
    #print(filtered)
    return filtered, entitiy_to_role

import string
# Step 3: Clean Data
def clean_data(data, file_path):
    """
    Final cleaning of the data. Removes brackets, index numbers, collections of points, XML markup and
    newlines in a row.
    :param data: String
    :param file_path: String
    :return: String
    """
    unprintable_reg = re.compile('[^%s]' % re.escape(string.printable))
    unprintable_reg.sub(data, "")
    data = re.sub('<.+\>', "", data)  # replaces xml markup
    data = re.sub('[()\[\]{}]', "", data) # replace brackets
    data = re.sub('[¹²³⁴⁵⁶⁷⁸⁹₁₂₃₄₅₆₇₈₉]', "", data) # replace index numbers
    data = re.sub('\.\.\.', ' ... ', data) # replaces e.g good...like
    data = re.sub("-(Member|Whole|Set)", " - ", data)
    data = re.sub("\n\n", "", data)
    data = re.sub("[:,;\"]", "", data)
    data = re.sub("!", " !", data)
    data = re.sub("\?", " ?", data)
    data = re.sub("(\w)\.", r"\1 . ", data)
    data = re.sub("/>", "", data)
    data = re.sub("\'", " '", data)
    data = re.sub("-", " - ", data)


    # write out data
    filename = file_path.split("/")[-1] # get the filename
    with open("D:/Korpora/cleaned_red/" + filename+".txt", "w+", encoding = "utf-8") as f:
        f.write(data)
    return data


if __name__ == "__main__":
    source = "dd0b65f632f64369c530f9bbb4b024b4.mpdf"
    xml = "dd0b65f632f64369c530f9bbb4b024b4.mpdf.RED-Relation.gold.completed.xml"
    with open(xml, "r", encoding="utf-8") as f:
        xml_data = f.read()
    with open(source, "r", encoding="utf-8") as f:
        data = f.read()
    print(data[1164:1167])
    #find_instances(xml)
    rel, ent = find_instances(xml) # extract relations and entities
    print(ent)
    print(rel)
    data = extract_from_source(rel, ent, source)
    data = clean_data(data)
    print(data)
