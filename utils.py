import subprocess

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class MultipleScheduler(object):
    def __init__(self, *sc):
        self.scheduler = sc

    def step(self):
        for sc in self.scheduler:
            sc.step()

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

DEFAULT_ATTRIBUTES = (
    'index',
    'memory.free',
    'memory.total'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def find_gpu():
    free_gpu = '0'
    free_memory = 0
    for gpu in get_gpu_info():
        if int(gpu['memory.total']) - int(gpu['memory.free']) < 100:
            return 'cuda:' + gpu['index']
        
        if int(gpu['memory.free']) > free_memory:
            free_memory = int(gpu['memory.free'])
            free_gpu = gpu['index']

    return 'cuda:'+free_gpu

