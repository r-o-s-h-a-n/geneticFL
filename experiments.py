import models


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
            population = self.model.generate_spawn(survivors)

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
            population = self.model.generate_spawn(parents) # S>C
            population_evals = self.model.evaluate_fitness(population) # C>C
            survivors = self.model.select_survivors(population_evals) # C>C
            parents = self.model.aggregate_survivors(survivors) # C>S

            if not i% 10:
                print(survivors, '\n')
                # get evaluations

        print(survivors)        
