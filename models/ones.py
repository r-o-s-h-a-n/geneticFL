from models.model import Model
import random
import numpy as np


class OnesModel(Model):
    '''
    Model that maximizes number of ones that appear in genetic sequence
    '''
    def __init__(self, ph):
        super(OnesModel).__init__(ph)

    def initialize(self):    
        population = [np.array([random.choice([0,1]) for i in range(self.gene_length)]) for _ in range(self.population_size)]
        return population

    def evaluate_fitness(self, population):
        population_evals = map(lambda x: (x, np.sum(x)), population)
        return population_evals

    def select_survivors(self, population_evals):
        survivors = sorted(population_evals, key=lambda x: x[1], reverse=True)[:2]
        return [x[0] for x in survivors]

    def crossover(self, parents):
        # uniform crossover
        child = np.empty(self.gene_length)
        for i in range(self.gene_length):
            child[i] = random.choice([p[i] for p in parents])
        return child

    def mutate(self, member):
        for i in range(GENE_LENGTH):
            member[i] = random.choices([member[i], not member[i]], 
                                    weights = [1-MUTATION_FRAC, MUTATION_FRAC],
                                    k = 1)[0]
        return member

    def generate_spawn(self, survivors):
        spawn = []
        for _ in range(NUM_CHILDREN):
            parents = random.choices(survivors, k=2)
            child = crossover(parents)
            child = mutate(child)
            spawn.append(child)
        return spawn

