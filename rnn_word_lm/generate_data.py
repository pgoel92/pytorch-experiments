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

nr_tokens = 10 
nr_training = 48000
nr_dev = 12000
nr_test = 15000

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
    nr_required_examples = nr_training + nr_dev + nr_test - count
    print nr_required_examples
    if nr_required_examples < len(examples):
        examples = examples[:nr_required_examples]
        print "B"
        done = True
    count += len(examples)
    with open('input.txt', 'a') as f:
        for e in examples:
            f.write(e + "\n")

final_nr_test = count / 5
final_nr_dev = (count - final_nr_test) / 5
test_command = 'cat input.txt | head -' + str(final_nr_test) + ' > test.txt' 
dev_command = 'cat input.txt | tail -' + str(final_nr_dev) + ' > validation.txt' 
train_command = 'cat input.txt | tail -' + str(count - final_nr_test - final_nr_dev) + ' > train.txt' 
os.system(test_command)
os.system(dev_command)
os.system(train_command)
os.system('rm -rf input.txt')
