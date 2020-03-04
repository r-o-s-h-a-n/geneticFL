import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(object):
    '''
    Your model must inherit from Model and specify the following methods:
        initialize
        evaluate_fitness
        select_survivors
        generate_spawn
    '''
    def __init__(self, ph):
        self.ph = ph
        self.gene_length = self.ph['gene_length']
        self.num_parents = self.ph['num_parents']
        self.num_survivors = self.ph['num_survivors']
        self.num_children = self.ph['num_children']
        self.mutation_frac = self.ph['mutation_frac']