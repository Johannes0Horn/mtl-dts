import argparse
import json
from collections import OrderedDict

import spacy
from tqdm import tqdm

MAPPING = {'bisabolol-': ['bisabolol', '-']}


def join_list(join_tokens, lis):
    joint_list = []

    for i, item in enumerate(lis):
        if item:
            joint_list.append(item)

        if i != len(lis) - 1:
            joint_list.extend(join_tokens)

    return joint_list


def prep_tokens(tokens):
    # corner case handling
    prepped_tokens = []

    for token, idx in tokens:
        prep_token = token

        if ')-' in token:
            prep_token = join_list([')', '-'], token.split(')-'))
        elif token in MAPPING:
            prep_token = MAPPING[token]

        if type(prep_token) == list:
            new_indices = []
            offset = 0
            for sp in prep_token:
                new_indices.append(idx + offset)
                offset += len(sp)
            assert (len(prep_token) == len(new_indices))
            prepped_tokens.extend(list(zip(prep_token, new_indices)))
        else:
            prepped_tokens.append((prep_token, idx))

    return prepped_tokens


def distance(c1, c2):
    return c2[0] - c1[1]


def find_entity(tokens, indices, entity_text, nlp):
    entity_tokens = nlp(entity_text)
    entity_tokens = [(t.text, t.idx) for t in entity_tokens]
    entity_tokens = [t[0] for t in prep_tokens(entity_tokens)]

    for i in range(len(tokens) - (len(entity_tokens) - 1)):
        if tokens[i:i + len(entity_tokens)] == entity_tokens:
            yield i, i + len(entity_tokens), indices[i], indices[i + len(entity_tokens) - 1] + len(
                tokens[i + len(entity_tokens) - 1])


def find_pair(tokens, indices, ae_text, drug_text, dist, trial, nlp):
    curr_trial = 0

    for head_start, head_end, h_sidx, h_eidx in find_entity(tokens, indices, ae_text, nlp):
        for tail_start, tail_end, t_sidx, t_eidx in find_entity(tokens, indices, drug_text, nlp):
            entity_dist = distance((h_sidx, h_eidx), (t_sidx, t_eidx))

            if dist == entity_dist:
                if curr_trial == trial:
                    return head_start, head_end, tail_start, tail_end
                curr_trial += 1

    assert False


def parse_sentence(sentence, nlp):
    add_dot = False
    if sentence[-1] == '.':
        sentence = sentence[:-1]
        add_dot = True

    tokens = nlp(sentence)
    tokens = [(t.text, t.idx) for t in tokens]
    tokens = prep_tokens(tokens)

    parsed_tokens = []
    parsed_indices = []  # token start indices

    for token, idx in tokens:
        if token.strip():
            parsed_tokens.append(token)
            parsed_indices.append(idx)

    if add_dot:
        parsed_indices.append(parsed_indices[-1] + len(parsed_tokens[-1]))
        parsed_tokens.append('.')

    return parsed_tokens, parsed_indices


def strip_entities(e_text, e_char_span):
    start, end = e_char_span

    if e_text != e_text.lstrip():
        start += len(e_text) - len(e_text.lstrip())

    if e_text != e_text.rstrip():
        end -= len(e_text) - len(e_text.rstrip())

    return e_text.strip(), (start, end)


def assign_id(assigned_ids, id_count, orig_doc_id, sentence):
    key = orig_doc_id + '_' + sentence.strip()

    if key not in assigned_ids:
        if orig_doc_id not in id_count:
            id_count[orig_doc_id] = 0

        assigned_ids[key] = orig_doc_id + '_' + str(id_count[orig_doc_id])
        id_count[orig_doc_id] += 1

    return assigned_ids[key]


def read_docs(lines, nlp):
    documents = OrderedDict()
    assigned_ids, id_count = dict(), dict()

    for line in tqdm(lines):
        orig_doc_id, _,sentence  = [string.strip() for string in line.split(" ", 2)]

        doc_id = assign_id(assigned_ids, id_count, orig_doc_id, sentence)

        if doc_id not in documents:
            documents[doc_id] = parse_sentence(sentence, nlp)

        tokens, indices = documents[doc_id]

    final_docs = []
    for k in documents.keys():
        doc_tokens = documents[k][0]
        final_docs.append(dict(tokens=doc_tokens, ADE=False, orig_id=k))

    return final_docs


def convert(source_path, dest_path, spacy_model):
    lines = open(source_path).readlines()
    nlp = spacy.load(spacy_model)
    documents = read_docs(lines, nlp)
    json.dump(documents, open(dest_path, 'w'))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--source_path', type=str, help="Path to dataset")
    arg_parser.add_argument('--dest_path', type=str, help="Destination file path (JSON format)")
    arg_parser.add_argument('--spacy_model', type=str, default='en_core_web_sm', help="SpaCy model")

    args = arg_parser.parse_args()
    convert(args.source_path, args.dest_path, args.spacy_model)