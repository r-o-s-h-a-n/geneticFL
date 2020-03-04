from models.model import Model
import random
import numpy as np
import collections
import tensorflow as tf
import tensorflow_federated as tff
from functools import wraps

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class OnesModelCentral(Model):
    '''
    Model that maximizes number of ones that appear in genetic sequence
    '''
    def __init__(self, ph):
        super(OnesModelCentral, self).__init__(ph)

    def init_population(self):
        population = [np.random.randint(low=0,high=2,size=self.gene_length,dtype=np.int32)
                                for _ in range(self.num_children)]
        return population

    def evaluate_fitness(self, population):
        population_fitness = map(lambda x: (x, np.sum(x)), population)
        return population_fitness

    def select_survivors(self, population_fitness):
        survivors = sorted(population_fitness, key=lambda x: x[1], reverse=True)[:self.num_parents]
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
            parent_idxs = random.sample(range(self.num_parents), k=2)
            child = self.crossover((survivors[parent_idxs[0]], survivors[parent_idxs[1]]))
            child = self.mutate(child)
            spawn.append(child)
        return spawn

    def aggregate_survivors(self, survivors):
        return survivors


def wrap_tff_federated_computation(spec_name, loc):
    def _wrap_tff_federated_computation(f):
        def wrapper(self, *args):
            spec = getattr(self, spec_name)

            @tff.federated_computation(tff.FederatedType(spec, loc))
            def g(*args):
                return f(self, *args)
            
            return g 
        return wrapper
    return _wrap_tff_federated_computation


class OnesModelFederated(OnesModelCentral):
    def __init__(self, ph):
        super(OnesModelFederated, self).__init__(ph)

        self.PARENTS_SPEC = tff.to_type(collections.OrderedDict([
            ('gene', tf.TensorSpec(shape=[self.num_parents, self.gene_length], dtype=tf.int32))
        ]))

        self.SURVIVORS_SPEC = tff.to_type(collections.OrderedDict([
            ('gene', tf.TensorSpec(shape=[self.num_survivors, self.gene_length], dtype=tf.int32))
        ]))

        self.CHILDREN_SPEC = tff.to_type(collections.OrderedDict([
            ('gene', tf.TensorSpec(shape=[self.num_children, self.gene_length], dtype=tf.int32))
        ]))

        self.FITNESS_SPEC = tff.to_type(collections.OrderedDict([
            ('gene', tf.TensorSpec(shape=[self.num_children, self.gene_length], dtype=tf.int32)),
            ('score', tf.TensorSpec(shape=[self.num_children], dtype=tf.float32))
        ]))

    def init_population(self):
        population = np.random.randint(low=0,high=2,size=(self.num_parents, self.gene_length),dtype=np.int32)
        return population

    @wrap_tff_federated_computation('PARENTS_SPEC', tff.SERVER)
    def generate_spawn(self, parents):

        @tff.tf_computation(self.PARENTS_SPEC)
        # @tf.function
        def _generate_spawn(local_parents):
            return local_parents
            
            # return tf.stack([local_parents['gene']
            #                 , local_parents['gene']
            #                 , local_parents['gene']
            #                 , local_parents['gene']
            #                 , local_parents['gene']
            #                 ]
            #                 , axis=0)
            # for x in local_survivors:
            #     tf.print(x)
            # # print(local_survivors)
            # # return survivors
            # spawn = tf.stack([local_survivors for _ in range(self.num_children//self.num_parents)], axis=0)
            # return spawn

        return tff.federated_map(_generate_spawn, tff.federated_broadcast(parents))

    @wrap_tff_federated_computation('CHILDREN_SPEC', tff.CLIENTS)
    def evaluate_fitness(self, population):
        
        @tff.tf_computation(self.CHILDREN_SPEC)
        def _evaluate_fitness(local_population):
            return (local_population['gene'], tf.math.reduce_sum(local_population['gene'], axis=1))

        return tff.federated_map(_evaluate_fitness, population)

    @wrap_tff_federated_computation('FITNESS_SPEC', tff.CLIENTS)
    def select_survivors(self, population_fitness):

        # def _np_select_survivors(population_fitness):
        #     survivors = sorted(population_fitness, key=lambda x: x[1], reverse=True)[:self.num_parents]
        #     return [x[0] for x in survivors]

        @tff.tf_computation(self.FITNESS_SPEC)
        # @tf.function
        def _select_survivors(population_fitness):
            return population_fitness['gene'][:self.num_survivors,:]
            # return population_fitness.take(1).map(lambda x: x['gene'])
            # return tf.numpy_function(_np_select_survivors, [population_fitness], [tf.int32, tf.float32])

        return tff.federated_map(_select_survivors, population_fitness)

    @wrap_tff_federated_computation('SURVIVORS_SPEC', tff.CLIENTS)
    def aggregate_survivors(self, survivors):
        # agg_survivors = tff.federated_collect(survivors)
        agg_survivors = tff.federated_sum(survivors)


        # print('aaa')

        # @tff.tf_computation(tff.SequenceType(self.SURVIVORS_SPEC))
        # @tf.function
        # def _agg_survivors(server_agg_survivors):
        #     return tf.stack(list(iter(server_agg_survivors.take(5))), axis=0)

        # print('bbb')
        # parents = tff.federated_map(_agg_survivors, agg_survivors)
        # print('ccc')
        # parents = _agg_survivors(agg_survivors)
        print(agg_survivors)
        return agg_survivors

    # def crossover(self, parents):
    #     # uniform crossover
    #     # child = collections.OrderedDict([
    #     #     ('genes', np.array([random.choice(g) for g in zip(parents.values())]))
    #     # ])
    #     print(parents)
    #     tf.print(parents)

    #     child = collections.OrderedDict([
    #         ('genes', tf.reduce_max(parents['genes'], axis=0))
    #     ])

        # child = {'gene': np.array([random.choice(g) for g in zip(parents.values())])}

        # child = np.empty(self.gene_length)
        # for i in range(self.gene_length):
        #     # child[i] = random.choice(parents.values())
        #     child[i] = random.choice([p[i] for p in parents])
        # return child



        #                         # reduce(np.zeros(self.gene_length), crossover)
        #                         # map(self.crossover)

        #         # parent_idxs = random.sample(range(self.num_parents), k=2)
        #         # # child = tf.py_function(random_rotate_image, [survivors], [tf.float32])
        #         # child = self.crossover((survivors[parent_idxs[0]], survivors[parent_idxs[1]]))
        #         # # child = self.crossover(**np.random.choice(survivors, size=2))
        #         # spawn.append(child)

# class OnesModelFederated(OnesModelCentral):
#     def __init__(self, ph):
#         super(OnesModelFederated, self).__init__(ph)

#         GENE_SPEC = collections.OrderedDict([
#             ('genes', tf.TensorSpec(shape=[self.gene_length], dtype=tf.int32))
#         ])
#         GENE_TYPE = tff.to_type(GENE_SPEC)

#         self.POPULATION_SPEC = tff.SequenceType(GENE_TYPE)
#         self.FITNESS_SPEC = tff.SequenceType(tff.NamedTupleType([('member', tf.float32), ('score', tf.float32)]))

#     def init_population(self):
#         population = [np.random.randint(low=0,high=2,size=self.gene_length,dtype=np.int32)
#                         for _ in range(self.num_parents)]
#         # population = [{'genes': np.random.randint(low=0,high=2,size=self.gene_length,dtype=np.int32)}
#         #                 for _ in range(self.num_parents)]
#         return population

#     @wrap_tff_federated_computation('POPULATION_SPEC', tff.SERVER)
#     def generate_spawn(self, survivors):

#         @tff.tf_computation(self.POPULATION_SPEC)
#         def _generate_spawn(survivors):
#             spawn = survivors.repeat(2*self.num_children//self.num_parents + 1).take(self.num_children*2).shuffle(
#                                 50).batch(2).take(1)
#             return spawn

#         return tff.federated_map(_generate_spawn, tff.federated_broadcast(survivors))

#     @wrap_tff_federated_computation('POPULATION_SPEC', tff.CLIENTS)
#     def evaluate_fitness(self, population):
#         @tff.tf_computation(self.POPULATION_SPEC)
#         def _evaluate_fitness(local_population):
#             return local_population.map(lambda x: (x['genes'], tf.math.reduce_sum(x['genes'])))
#         return tff.federated_map(_evaluate_fitness, population)

#     @wrap_tff_federated_computation('FITNESS_SPEC', tff.CLIENTS)
#     def select_survivors(self, population_fitness):
#         survivors = sorted(population_fitness, key=lambda x: x['score'], reverse=True)[:self.num_parents]
#         survivors = [x[0] for x in survivors]
#         return survivors

#     @wrap_tff_federated_computation('POPULATION_SPEC', tff.CLIENTS)
#     def aggregate_survivors(self, survivors):
#         return survivors