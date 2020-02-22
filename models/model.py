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
        self.population_size = self.ph['population_size']
        self.num_children = self.ph['num_children']
        self.mutation_frac = self.ph['mutation_frac']