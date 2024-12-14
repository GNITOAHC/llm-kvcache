import sys
import random

global seed

def randtest(target):
    if seed:
        random.seed(seed)
        random.shuffle(target["arr"])
        random.shuffle(target["arr2"])
    
if __name__ == '__main__':
    seed = sys.argv[1] if len(sys.argv) > 1 else None
    
    data = {
        "arr": [i * 100 for i in range(10)],
        "arr2": [i for i in range(10)]
    }
    randtest(data)
    
    print(data)