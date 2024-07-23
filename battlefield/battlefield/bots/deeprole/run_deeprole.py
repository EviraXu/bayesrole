import subprocess
import json
import os
import numpy as np

from battlefield.bots.deeprole.lookup_tables import ASSIGNMENT_TO_VIEWPOINT
from nn_train.train import predict

DEEPROLE_BASE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'deeprole')
DEEPROLE_BINARY_BASE = os.path.join(DEEPROLE_BASE_DIR, 'code')


deeprole_cache = {}

def actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder, binary):
    command = [
        os.path.join(DEEPROLE_BINARY_BASE, binary),
        '--play',
        '--proposer={}'.format(node['proposer']),
        '--succeeds={}'.format(node['succeeds']),
        '--fails={}'.format(node['fails']),
        '--propose_count={}'.format(node['propose_count']),
        '--depth=1',
        '--iterations={}'.format(iterations),
        '--witers={}'.format(wait_iterations),
        '--modeldir={}'.format(nn_folder)
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEEPROLE_BASE_DIR
    )
    if no_zero:
        belief = node['nozero_belief']
    else:
        belief = node['new_belief']
    #print(str(belief))
    stdout, stderr = process.communicate(input=(str(belief) + "\n").encode('utf-8'))
    #print("Stdout:", stdout)
    #print("Stderr:", stderr.decode('utf-8'))

    result = json.loads(stdout)
    return result

def actually_run_deeprole_assignment_on_node(node, iterations, wait_iterations, history, no_zero, nn_folder, binary):
    command = [
        os.path.join(DEEPROLE_BINARY_BASE, binary),
        '--play',
        '--proposer={}'.format(node['proposer']),
        '--succeeds={}'.format(node['succeeds']),
        '--fails={}'.format(node['fails']),
        '--propose_count={}'.format(node['propose_count']),
        '--depth=1',
        '--iterations={}'.format(iterations),
        '--witers={}'.format(wait_iterations),
        '--modeldir={}'.format(nn_folder)
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=DEEPROLE_BASE_DIR
    )
    if no_zero:
        belief = node['nozero_belief']
    else:
        #belief = node['new_belief']
        #如果history全是-1，则令belief60个值每个等于1/60
        if (history == -1).all():
            # 如果全部是 -1，则令 belief 的 60 个值每个等于 1/60
            belief = [1.0 / 60] * 60
        else:
            # 否则使用 predict(history) 的返回值
            belief = predict(history)
    #print("history:",history)
    #print("belief:",belief)

    def convert_belief_to_string(belief):
        belief_list = belief.tolist()  # 将numpy数组转换为Python列表
        belief_str = '[' + ', '.join(f"{num:.8e}" for num in belief_list) + ']'
        return belief_str
    if (history == -1).all():
        belief_str = belief
    else:
        belief_str = convert_belief_to_string(belief[0])
    
    #print("belief_str:",belief_str)
    stdout, stderr = process.communicate(input=(str(belief_str) + "\n").encode('utf-8'))
    #print("Stdout:", stdout)
    #print("Stderr:", stderr.decode('utf-8'))
    # try:
    #     # 将 stdout 内容保存为 JSON 文件
    #     with open("node.json", 'w') as f:
    #         f.write(stdout.decode('utf-8'))
    # except Exception as e:
    #     print(f"Error writing JSON file: {e}")

    result = json.loads(stdout)
    return result


def run_deeprole_on_node(node, iterations, wait_iterations, no_zero=False, nn_folder='deeprole_models', binary='deeprole'):
    #print("--run_deeprole_on_node---")
    global deeprole_cache

    if len(deeprole_cache) > 250:
        deeprole_cache = {}

    cache_key = (
        node['proposer'],
        node['succeeds'],
        node['fails'],
        node['propose_count'],
        tuple(node['new_belief']),
        iterations,
        wait_iterations,
        no_zero,
        nn_folder,
        binary
    )
    
    if cache_key in deeprole_cache:
        return deeprole_cache[cache_key]

    result = actually_run_deeprole_on_node(node, iterations, wait_iterations, no_zero, nn_folder, binary)
    deeprole_cache[cache_key] = result
    return result



def run_deeprole_assignment_on_node(node, iterations, wait_iterations, history, no_zero=False, nn_folder='deeprole_models', binary='deeprole'):
    print("--run_deeprole_assignment_on_node---")
    global deeprole_cache

    if len(deeprole_cache) > 250:
        deeprole_cache = {}
    
    #print(history)

    def convert_to_tuple(item):
        if isinstance(item, np.ndarray):
            return tuple(map(convert_to_tuple, item))
        elif isinstance(item, list):
            return tuple(map(convert_to_tuple, item))
        else:
            return item

    # Convert history to a hashable tuple
    history_tuple = convert_to_tuple(history)

    cache_key = (
        node['proposer'],
        node['succeeds'],
        node['fails'],
        node['propose_count'],
        tuple(node['new_belief']),
        iterations,
        wait_iterations,
        history_tuple,
        no_zero,
        nn_folder,
        binary
    )
    
    if cache_key in deeprole_cache:
        return deeprole_cache[cache_key]

    result = actually_run_deeprole_assignment_on_node(node, iterations, wait_iterations, history, no_zero, nn_folder, binary)
    deeprole_cache[cache_key] = result
    return result
