import models


class Algorithm(object):
    def __init__(self, ph):
        self.ph = ph
        self.model = getattr(models, self.ph['model'])(self.ph)
        self.num_generations = self.ph['num_generations']

    def run(self):
        population = self.model.initialize()
        
        for i in range(self.num_generations):
            population_evals = self.model.evaluate_fitness(population)
            survivors = self.model.select_survivors(population_evals)
            population = self.model.generate_spawn(survivors)

            if not i% 10:
                print(survivors, '\n')

        print(survivors)


class CentralAlgorithm(Algorithm):
    def __init__(self, ph):
        super(CentralAlgorithm).__init__()


class FederatedAlgorithm(Algorithm):
    def __init__(self, ph):
        super(FederatedAlgorithm).__init__()
