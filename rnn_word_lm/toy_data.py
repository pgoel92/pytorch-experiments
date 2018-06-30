import random

vocab = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
with open('toy_train.txt', 'w') as f:
    for i in range(50000):
        ex = ' '.join([ vocab[random.randint(0, 25)] for j in range(3) ])
        f.write(ex + "\n")

