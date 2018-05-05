import random

vocab = ['a','b','c','d','e'];
with open('toy_train.txt', 'w') as f:
    for i in range(10000):
        ex = ' '.join([ vocab[random.randint(0, 4)] for j in range(3) ])
        f.write(ex + "\n")

