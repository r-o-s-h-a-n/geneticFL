from models.model import Model
import random
import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff
from functools import wraps


class OnesModelCentral(Model):
    '''
    Model that maximizes number of ones that appear in genetic sequence
    '''
    def __init__(self, ph):
        super(OnesModelCentral, self).__init__(ph)

    def init_population(self):
        population = [np.random.randint(low=0,high=2,size=self.gene_length,dtype=np.int32)
                                for _ in range(self.population_size)]
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
        for i in range(self.gene_length):
            member[i] = np.random.choice([member[i], not member[i]], 
                                    p = [1-self.mutation_frac, self.mutation_frac],
                                    )
        return member

    def generate_spawn(self, survivors):
        spawn = []
        for _ in range(self.num_children):
            parent_idxs = random.sample(range(len(survivors)), k=2)
            child = self.crossover((survivors[parent_idxs[0]], survivors[parent_idxs[1]]))
            child = self.mutate(child)
            spawn.append(child)
        return spawn


def wrap_tff_federated_computation(method):
    def wrapper(*args):
        self = args[0]
        return f(*args)
    return wrapper

class OnesModelFederated(OnesModelCentral):
    def __init__(self, ph):
        super(OnesModelFederated, self).__init__(ph)

        GENE_SPEC = collections.OrderedDict([
            ('genes', tf.TensorSpec(shape=[self.gene_length], dtype=tf.int32))
        ])

        GENE_TYPE = tff.to_type(GENE_SPEC)

        self.POPULATION_SPEC = tff.SequenceType(GENE_TYPE)

    def init_population(self):
        population = [{'genes': np.random.randint(low=0,high=2,size=self.gene_length,dtype=np.int32)}
                        for _ in range(self.population_size)]
        return population

    @tff.federated_computation(tff.SequenceType(self.GENE_SPEC), tff.SERVER)
    def generate_spawn(self, survivors):
        survivors = tff.federated_broadcast(survivors)
        spawn = []
        for _ in range(self.num_children):
            parent_idxs = random.sample(range(len(survivors)), k=2)
            child = self.crossover((survivors[parent_idxs[0]], survivors[parent_idxs[1]]))
            spawn.append(child)
        return spawn

    @tff.tf_computation(tff.SequenceType(self.GENE_SPEC), tff.CLIENT)
    def evaluate_fitness(self, population):
        population_evals = None
        return population_evals

    @tff.tf_computation(tff.SequenceType(self.GENE_SPEC), tff.CLIENT) # this one is wrong, should be tuple
    def select_survivors(self, population_evals):
        survivors = None
        return survivors

    @tff.federated_computation(tff.SequenceType(self.GENE_SPEC), tff.CLIENT)
    def aggregate_survivors(self, survivors):
        parents = None
        return parents