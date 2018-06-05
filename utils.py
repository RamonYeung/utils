"""
Notes: this utils is tested on MacOS and Ubuntu only. I guess it is not supported by Windows.
"""
import functools
import os
import operator
import random
import subprocess
import time
import torch
import torch.nn as nn
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from unidecode import unidecode
# assert nltk.__version__ == '3.2.4'


def estimate_cuda_memory_usage(model: nn.Module, *inputs, over_estimate=True):
    """
    Statistics on model parameters and memory usage.
    """
    dtype2bytes = {torch.uint8: 1,
                   torch.float: 4, torch.float32: 4,
                   torch.float64: 8, torch.double: 8,
                   torch.int64: 8, torch.long: 8
                   }

    # Bytes to megabytes
    scale = 1000 ** 2 if over_estimate else 1024 ** 2
    num_param = 0
    mem_usage = 0
    print(f'Estimating Memory Usage for model: {type(model).__name__} ...\n')
    print(f"{' Layer ':=^16s}{' #Params ':=>13s}{' Memory usage(MB) ':=^24s}")

    # Memory usage for inputs
    assert all(inputs[0].shape[0] == tensor.shape[0] for tensor in inputs)
    for tensor in inputs:
        param_per_tensor = functools.reduce(operator.mul, tensor.shape) * dtype2bytes[tensor.dtype]
        usage_per_tensor = param_per_tensor / scale
        num_param += param_per_tensor
        mem_usage += usage_per_tensor
    print(f"{'INPUT': <16s}{num_param: >12,d}{mem_usage: >15.3f}")

    # Memory usage for layers
    for name, param in model.named_parameters():
        param_per_layer = functools.reduce(operator.mul, param.shape) * dtype2bytes[param.dtype]
        usage_per_layer = param_per_layer / scale * (2 if param.requires_grad else 1)

        num_param += param_per_layer
        mem_usage += usage_per_layer
        print(f"{name: <16s}{param_per_layer: >12,d}{usage_per_layer: >15.3f}")

    # Memory usage for outputs
    try:
        out_dim = model.out_dim
        param_output = inputs[0].shape[0] * out_dim
        usage_output = param_output / scale

        num_param += param_output
        mem_usage += usage_output
        print(f"{'OUTPUT': <16s}{param_output: >12,d}{usage_output: >15.3f}")

    except AttributeError as e:
        print(e)
        print("Please implement 'self.out_dim' for models")
        os.system('kill -9 %d' % os.getpid())

    print(f"{'TOTAL': <16s}{num_param: >12,d}{mem_usage: >15.4f}\n")

    return mem_usage


def get_available_device(expected_capacity=8096):
    """
    It is boring that one has to find available devices every time training or testing models.
    Numbers are in MegaBytes.
    TODO batch_size * 3 memory usage
    """
    cuda_context = 511  # CUDA consumes memory when init context states of CUDA.

    cmd = ['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,nounits,noheader']
    result = subprocess.check_output(cmd)

    free_memory_stats = [line.split(', ') for line in result.decode('utf-8').strip().split('\n')]
    for device_id, name, free_memory in free_memory_stats:
        if int(free_memory) > expected_capacity + cuda_context:
            yield device_id, name, int(free_memory)


def gen_candidate_entity(question):
    """
    Given a question, generate candidate entities according length of LCS between the question and the entity.
    o accelerate this algorithm, stop words in question are filtered out.
    """
    # stop_words = load_stop_word(None)
    # print(question)
    # question = ' '.join([word for word in question.split() if word not in stop_words])
    # print(question)

    candidate_entity = []
    id2entity = load_entity_dict(file_path='./data/kb_entity_dict.txt')

    for entity in id2entity:
        seq_match = SequenceMatcher(None, entity, question)
        # find match of longest sub-string, output will be like Match(a=0, b=0, size=5)
        match = seq_match.find_longest_match(0, len(entity), 0, len(question))
        llcs = match.size
        if llcs / len(entity) > 0.67:  # more than 2/3 matched
            candidate_entity.append(entity)
    print(f'#candidate_entity for {question}: {candidate_entity}')


def load_single_fact(file_path='./data/kb_1_hop.txt'):
    single_fact_pool = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = normalize(line)
            s, p, o = line.split('||')
            single_fact_pool[s] = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = normalize(line)
            s, p, o = line.split('||')
            single_fact_pool[s][p] = o
    return single_fact_pool


def load_single_fact_as_table(file_path='./data/kb_1_hop.txt'):
    single_fact_pool = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = normalize(line)
            s, p, o = line.split('||')
            single_fact_pool[s + ' ' + p] = o
    return single_fact_pool


def load_entity_dict(file_path='./data/kb_entity_dict.txt'):
    """id2entity is a list of list if an entity contains multiple words separated by black. """
    id2entity = set()
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = normalize(line)
            id2entity.add(line.split('\t')[1])
    return id2entity


def load_stop_word(file_path='./data/stop_words.txt'):
    if file_path:
        stop_words = set()
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
    else:
        start = time.time()
        # NLTK stop words
        stop_words = set(stopwords.words('english'))

        # Load all questions in train and dev
        counter = Counter()
        with open('./data/train.json.txt', encoding='utf-8') as f:
            for line in f:
                line = normalize(line)
                question = line.split('||', 1)[0]
                tokens = question.split()
                counter.update(tokens)
        with open('./data/dev.json.txt', encoding='utf-8') as f:
            for line in f:
                line = normalize(line)
                question = line.split('||', 1)[0]
                tokens = question.split()
                counter.update(tokens)

        # most common (top 0.1%) words as stop words,
        most_common_words = list(map(list, zip(*counter.most_common(20))))[0]
        print(f'#total words: {len(counter)}. ({time.time() - start:.4f} seconds)')

        stop_words = frozenset(stop_words | set(most_common_words))

    return stop_words


def normalize(s):
    s = s.strip().lower()
    s = unidecode(s)
    if '_' in s:
        s = s.replace('_', ' ')
    return s


def average_precision(k_true, k_score):
    assert k_true.shape == k_score.shape

    # cumulate sum
    # 1 1 2 2 3 4 5
    # 1 2 3 4 5 6 7
    correct, total = 0, 0
    precision = []
    for ground_truth, predicted in zip(k_true, k_score):
        total += 1
        if ground_truth == predicted:
            correct += 1
        precision.append(correct / total)

    ap = sum(precision) / len(precision)
    return ap


def batch(dataset, batch_size):
    num_data_points = len(dataset)
    for ndx in range(0, num_data_points, batch_size):
        yield dataset[ndx: ndx + batch_size]  # IndexError handled build-in.


def set_seed(seed):
    """
    Freeze every seed.
    All about reproducibility
    TODO multiple GPU seed, torch.cuda.all_seed()
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
