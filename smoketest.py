'''
Smoke test script to check packages are installed and models are available locally
'''

import sys
print(f'''
Python interpreter: {sys.executable}
Python version:     {sys.version}
''')

import huggingface_hub as hfh
print(f'huggingface_hub: {hfh.__version__}')

import deepeval as de
print(f'deepeval:        {de.__version__}')

import evidently as ev
print(f'evidently:       {ev.__version__}')

import ollama
models = list(ollama.list())[0][1]

print('\n\nOllama models (local):\n')
for m in models:
    print(f'{m.model:<30}\t{m.details.family}\t{m.details.parameter_size}\t{int(m.size/(1e6)):>6} MB\t{m.details.quantization_level}\t{m.details.format}')
