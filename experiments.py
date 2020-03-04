import models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Algorithm(object):
    def __init__(self, ph):
        self.ph = ph
        self.model = getattr(models, self.ph['model'])(self.ph)
        self.num_generations = self.ph['num_generations']


class TraditionalGeneticAlgorithm(Algorithm):
    def __init__(self, ph):
        super(TraditionalGeneticAlgorithm, self).__init__(ph)

    def run(self):
        population = self.model.init_population()
        
        for i in range(self.num_generations):
            population_evals = self.model.evaluate_fitness(population)
            survivors = self.model.select_survivors(population_evals)
            parents = self.model.aggregate_survivors(survivors)
            population = self.model.generate_spawn(parents)

            if not i% 10:
                print(survivors, '\n')
                # get evaluations

        print(survivors)


class ModifiedGeneticAlgorithm(Algorithm):
    def __init__(self, ph):
        super(ModifiedGeneticAlgorithm, self).__init__(ph)

    def run(self):
        parents = self.model.init_population()

        for i in range(self.num_generations):
            print('\n\n\n\n\n{}\n\n\n\n\n'.format(str(i)))

            population = self.model.generate_spawn(parents) # S>C
            print('a')
            print(population)
            population_evals = self.model.evaluate_fitness(population) # C>C
            print('b')
            print(population_evals)
            survivors = self.model.select_survivors(population_evals) # C>C
            print('c')
            print(survivors)
            parents = self.model.aggregate_survivors(survivors) # C>S
            print('d')
            print(parents)


            if not i% 10:
                # print(parents())
                print(parents, '\n')
                # get evaluations

        print(survivors)        
