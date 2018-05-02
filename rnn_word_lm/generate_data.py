#!/usr/bin/python

import os

DATASET_PATH = '/Users/pgoel/Git/pytorch/data/1-billion-word-language-modeling-benchmark-r13output'
TRAINING_PATH = DATASET_PATH + '/training-monolingual.tokenized.shuffled'
HELDOUT_PATH = DATASET_PATH + '/heldout-monolingual.tokenized.shuffled'

def get_filename(i, heldout=''):
   file_serial_nr = '000' + str(i) if len(str(i)) == 2 else '0000' + str(i)
   path = HELDOUT_PATH if heldout else TRAINING_PATH
   return path + '/news.en' + heldout + '-'+ file_serial_nr +'-of-00100'

def get_training_filenames():
   return [get_filename(i) for i in range(100)]

def get_heldout_filenames():
   return [get_filename(i, 'heldout') for i in range(50)]

def get_examples_with_token_count(filename, nr_tokens):
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = map(lambda x: x.strip(), lines)
        token_lines = [line for line in lines if len(line.split()) == nr_tokens]
        return token_lines

nr_tokens = 5
nr_training = 50000
nr_dev = 100
nr_test = 500

training_files = get_training_filenames()
heldout_files = get_heldout_filenames()

i = 0
count = 0
done = False
while not done:
    filename = training_files[i]
    i += 1
    if i == 100:
        print "A"
        done = True
    examples = get_examples_with_token_count(filename, nr_tokens)
    nr_required_examples = nr_training + nr_dev - count
    print nr_required_examples
    if nr_required_examples < len(examples):
        examples = examples[:nr_required_examples]
        print "B"
        done = True
    count += len(examples)
    with open('input.txt', 'a') as f:
        for e in examples:
            f.write(e + "\n")

train_command = 'cat input.txt | head -' + str(nr_training) + ' > train.txt' 
dev_command = 'cat input.txt | tail -' + str(nr_dev) + ' > dev.txt' 
os.system(train_command)
os.system(dev_command)
os.system('rm -rf input.txt')
