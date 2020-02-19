from models.model import Model
import random
import numpy as np


class EMNISTModel(Model):
    '''
    Model that learns to classifies digits in EMNIST dataset
    '''
    def __init__(self):
        super(EMNISTModel).__init__()

    def initialize(self):    
        pass

    def evaluate_fitness(self, population):
        pass

    def select_survivors(self, population_evals):
        pass

    def crossover(self, parents):
        pass

    def mutate(self, member):
        pass

    def generate_spawn(self, survivors):
        pass