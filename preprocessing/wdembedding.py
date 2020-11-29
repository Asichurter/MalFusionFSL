import os

def trainGloVe(base_path, dataset, size=300, type='all'):
    print("Running GloVe in Python 2 env using Shell...")
    os.system(f'python2 ../utils/GloVe.py -p {base_path} -d {dataset} -s {size} -t {type}')
    print('GloVe Done')