import math
import random
import csv
import os

def do_crowding_distance(individuals, fixed = False):
    '''Crowding distance of Deb et al. (2002). The original implementation (fixed = False) adds the
    accumulated distance to itself on every objective pass, i.e. for two objectives it computes
    2 * d_1 + d_2 instead of d_1 + d_2; fixed = True computes the standard sum.'''
    m = len(individuals[0].fitness)
    n = len(individuals)

    # initialize crowding distances
    for i in range(n):
        individuals[i].cdistance = 0.0

    # determine min and max values for all objectives
    f_max = [max([ind.fitness[i] for ind in individuals]) for i in range(m)]
    f_min = [min([ind.fitness[i] for ind in individuals]) for i in range(m)]

    norm = [f_max[i] - f_min[i] for i in range(m)]
    # Workaround: sometimes we get an error because f_max[i] = f_min[i] and thus be have a zero division
    for i in range(m):
        if norm[i] < 0.001:
            norm[i] = 1

    for i in range(m):
        individuals.sort(key = lambda e: e.fitness[i])
        individuals[0].cdistance = float('inf')
        individuals[n - 1].cdistance = float('inf')
        for j in range(1, n - 1):
            if fixed:
                individuals[j].cdistance += (individuals[j + 1].fitness[i] - individuals[j - 1].fitness[i]) / norm[i]
            else:
                individuals[j].cdistance += individuals[j].cdistance + (individuals[j + 1].fitness[i] - individuals[j - 1].fitness[i]) / norm[i]

    return individuals

def dominates(x, y):
    n = len(x)
    assert len(x) == len(y)
    return all([x[i] <= y[i] for i in range(n)]) and any([x[i] < y[i] for i in range(n)])

def do_fast_nondominated_sorting(individuals):
    """
    Fast non-dominated sorting algorithm proposed by Deb.

    Non-dominated sorting expects a set of points and returns a set of non-dominated layers. In short
    words this is done as follows: the non-dominated points of the entire set
    are determined and assigned rank 1. Afterwards all points with the current
    rank are removed, the rank is increased by one and the procedure starts again.
    This is done until the set is empty, i.e., each point is assigned a rank.

    Deb, K., Pratap, A., and Agarwal, S. A Fast and Elitist Multiobjective Genetic
    Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6 (8) (2002),
    182-197.
    """
    layers = [[]]
    layers_indices = [[]]

    n = len(individuals)

    # stores the number of individuals by which a point is dominated
    dom_counter = [0] * n

    # non-domination rank
    ranks = [0] * n

    # list of point IDs a point dominates
    dom_elements = [[] for _ in range(n)]

    # iterate pairs of individuals and check dominance relation
    for i in range(n):
        for j in range(n):
            if dominates(individuals[i].fitness, individuals[j].fitness):
                dom_elements[i].append(j)
            elif dominates(individuals[j].fitness, individuals[i].fitness):
                dom_counter[i] += 1

        # all non-dominated individuals are assigned rank 1
        if dom_counter[i] == 0:
            ranks[i] = 1
            layers[0].append(individuals[i])
            layers_indices[0].append(i)

    # now determine the remaining ranks
    k = 0
    while len(layers_indices[k]) > 0:
        layer2 = []
        layer2_indices = []
        for i in layers_indices[k]:
            for j in dom_elements[i]:
                dom_counter[j] = dom_counter[j] - 1
                if dom_counter[j] == 0:
                    ranks[j] = k + 2
                    layer2.append(individuals[j])
                    layer2_indices.append(j)

        k += 1
        layers.append(layer2)
        layers_indices.append(layer2_indices)

    return layers


class NSGA2Individual(object):

    def __init__(self, x):
        """
        NSGA-II individual object.

        Fields:
            x [any]: the individual itself (whatever encoding is used).
            fitness [list[float]]: the individuals' fitness vector.
            cdistance float: the individuals' crowding distance.
        """
        super().__init__()
        self.x = x
        self.fitness = None
        self.cdistance = 0.0

    def evaluate(self, objective_fun):
        """
        Calculate the individuals' objective function value.
        """

        if self.fitness is None:
            self.fitness = objective_fun(self.x)
        return self.fitness

    def get_solution(self):
        return self.x

    def get_fitness(self):
        return self.fitness


class NSGA2(object):

    def __init__(self,
        mu: int,
        lambdaa: int,
        initializer,
        objective_fun,
        max_evals: int,
        mutator = None,
        crossover = None,
        p_crossover: float | None = None,
        p_c: float | None = None,
        p_m: float | None = None,
        log_every: int | None = 10,
        on_generation = None,
        distinct_parents: bool = False,
        discard_clones: bool = False,
        max_attempts_factor: int = 100,
        fixed_crowding_distance: bool = False,
        eliminate_duplicates: bool = False) -> None:
        """
        Simplified NSGA-II. This implementation deviates from the NSGA-II by Deb et al. in the following two points:
        * Parent selection is uniformly at random (i.e., not based on binary tournament selection and the partial
          crowding distance order).
        * It allows for a (mu + lambda) strategy with arbitrary integer lambda > 0.

        Reproduction -- three mutually exclusive ways to combine a crossover and a mutation operator,
        selected by which of p_crossover / (p_c, p_m) is given:

        * p_crossover ("either/or"): each offspring is generated by crossover of two randomly sampled
          parents with probability p_crossover, and by mutation of a single randomly sampled parent
          with probability 1 - p_crossover. Never both for the same offspring.
        * p_c and p_m ("independent Bernoulli trials"): for each offspring, a base individual is first
          obtained by crossover of two randomly sampled parents with probability p_c, or otherwise (prob.
          1 - p_c) by copying a single randomly sampled parent unchanged. That base is then, independently,
          mutated with probability p_m (left as-is otherwise). So an offspring can end up as crossover-only,
          crossover-then-mutation, mutation-of-a-single-parent-only, or an untouched copy of one parent --
          unlike p_crossover, where exactly one of crossover/mutation always happens, never both, never
          neither.
        * Neither given: if a crossover operator is given, offspring are generated by crossing two randomly
          sampled parents; the result is additionally mutated if a mutator is given (i.e. equivalent to
          p_c = 1.0, p_m = 1.0 if both operators are given). If no crossover operator is given, offspring
          are generated by mutating a single randomly sampled parent (mutator required in that case).

        on_generation (optional): observer called as on_generation(gen, parents, offspring, survivors)
        once for the initial population (gen = 0, parents = offspring = None) and after every survival
        selection (parents = P_{gen-1}, offspring = Q_gen, survivors = P_gen). It must not consume
        random numbers, so a run with an observer stays identical to one without.

        Duplicate handling for crossover (both off by default, which keeps the original sampling and
        therefore the original random number stream; used for crossover-only runs and p_crossover):
        * distinct_parents: the second parent is redrawn until its tree (edge set) differs from the first
          parent's tree. If all trees in the population are identical, the pair is accepted as drawn.
        * discard_clones (requires distinct_parents): an offspring identical to one of its two parents is
          discarded without evaluation and does not count towards the lambda offspring; a new parent pair
          is drawn instead. After max_attempts_factor * lambda crossover calls in one generation, clones
          are accepted to guarantee termination.
        Counters are collected in self.repro_stats.

        Survival selection (both off by default, i.e. the original behaviour):
        * fixed_crowding_distance: standard crowding distance d_1 + ... + d_m instead of the original
          implementation's 2 * d_1 + d_2 (see do_crowding_distance).
        * eliminate_duplicates: before survival selection, every tree (edge set) that already occurs
          earlier in R = P + Q is removed (parents come first, so a clone of a surviving tree is the
          one removed). If fewer than mu distinct trees remain, removed duplicates are added back in
          their original order until R has mu elements.
        """
        super().__init__()

        assert mu >= 5
        assert lambdaa >= 1
        assert mutator is not None or crossover is not None
        if p_crossover is not None:
            assert p_c is None and p_m is None
            assert crossover is not None and mutator is not None
            assert 0.0 < p_crossover < 1.0
        if p_c is not None or p_m is not None:
            assert p_crossover is None
            assert p_c is not None and p_m is not None
            assert crossover is not None and mutator is not None
            assert 0.0 <= p_c <= 1.0
            assert 0.0 <= p_m <= 1.0
        if distinct_parents or discard_clones:
            assert crossover is not None and p_c is None and p_m is None
            assert p_crossover is not None or mutator is None
        if discard_clones:
            assert distinct_parents

        self.mu = int(mu)
        self.lambdaa = int(lambdaa)
        self.initializer = initializer
        self.objective_fun = objective_fun
        self.mutator = mutator
        self.crossover = crossover
        self.p_crossover = p_crossover
        self.p_c = p_c
        self.p_m = p_m
        self.max_evals = max_evals

        # calculate number of generations
        self.max_gens = math.ceil((self.max_evals - self.mu) / self.lambdaa)

        self.log_every = log_every
        self.on_generation = on_generation

        self.distinct_parents = distinct_parents
        self.discard_clones = discard_clones
        self.max_attempts_factor = max_attempts_factor
        self.fixed_crowding_distance = fixed_crowding_distance
        self.eliminate_duplicates = eliminate_duplicates
        self.duplicates_removed = 0
        self.repro_stats = dict(cx_calls = 0, parent_redraws = 0, no_distinct_partner = 0,
                                clones_discarded = 0, clones_accepted_at_cap = 0)

        # number of function evaluations
        self.evals = 0
        self.gens = 0

        # history
        self.pf = None
        self.history = []

    def sample_parents(self):
        return [random.randint(0, self.mu - 1) for _ in range(self.lambdaa)]

    def sample_parent_pairs(self):
        return [(random.randint(0, self.mu - 1), random.randint(0, self.mu - 1)) for _ in range(self.lambdaa)]

    @staticmethod
    def tree_key(tree):
        return frozenset((e.v, e.w) if e.v < e.w else (e.w, e.v) for e in tree.get_edges())

    def crossover_child(self, P, keys, attempts):
        '''One crossover offspring with the duplicate handling of distinct_parents / discard_clones.
        keys[i] is the edge set of P[i]; attempts counts crossover calls of this generation.'''
        st = self.repro_stats
        while True:
            i = random.randint(0, self.mu - 1)
            j = random.randint(0, self.mu - 1)
            if keys[j] == keys[i]:
                if all(k == keys[i] for k in keys):
                    st['no_distinct_partner'] += 1
                else:
                    while keys[j] == keys[i]:
                        st['parent_redraws'] += 1
                        j = random.randint(0, self.mu - 1)
            child = self.crossover(P[i].x, P[j].x)
            st['cx_calls'] += 1
            attempts[0] += 1
            if not self.discard_clones:
                return child
            ck = self.tree_key(child)
            if ck != keys[i] and ck != keys[j]:
                return child
            if attempts[0] >= self.max_attempts_factor * self.lambdaa:
                st['clones_accepted_at_cap'] += 1
                return child
            st['clones_discarded'] += 1

    def run(self) -> list:

        # initialise population
        P = [NSGA2Individual(self.initializer()) for _ in range(self.mu)]
        # print(f"Number of individuals in initial population: {len(P)}")

        # calculate fitness
        P_fitness = [x.evaluate(self.objective_fun) for x in P]
        self.evals = self.mu

        if self.on_generation is not None:
            self.on_generation(self.gens, None, None, P)

        if self.log_every is not None:
            if (self.gens % self.log_every) == 0:
                self.history.append(P[:])

        # main loop
        while self.gens < self.max_gens:
            # generate offspring
            # print(f"|P_{self.gens}| = {len(P)}")

            if self.distinct_parents:
                keys = [self.tree_key(ind.x) for ind in P]
                attempts = [0]
                children = []
                for _ in range(self.lambdaa):
                    if self.p_crossover is None or random.random() < self.p_crossover:
                        children.append(self.crossover_child(P, keys, attempts))
                    else:
                        i = random.randint(0, self.mu - 1)
                        children.append(self.mutator(P[i].x))
            elif self.p_crossover is not None:
                children = []
                for _ in range(self.lambdaa):
                    if random.random() < self.p_crossover:
                        i, j = random.randint(0, self.mu - 1), random.randint(0, self.mu - 1)
                        children.append(self.crossover(P[i].x, P[j].x))
                    else:
                        i = random.randint(0, self.mu - 1)
                        children.append(self.mutator(P[i].x))
            elif self.p_c is not None:
                children = []
                for _ in range(self.lambdaa):
                    if random.random() < self.p_c:
                        i, j = random.randint(0, self.mu - 1), random.randint(0, self.mu - 1)
                        child = self.crossover(P[i].x, P[j].x)
                    else:
                        i = random.randint(0, self.mu - 1)
                        child = P[i].x.copy()
                    if random.random() < self.p_m:
                        child = self.mutator(child)
                    children.append(child)
            elif self.crossover is not None:
                pairs = self.sample_parent_pairs()
                children = [self.crossover(P[i].x, P[j].x) for i, j in pairs]
                if self.mutator is not None:
                    children = [self.mutator(child) for child in children]
            else:
                parents = [P[i] for i in self.sample_parents()]
                children = [self.mutator(x.x) for x in parents]

            Q = [NSGA2Individual(child) for child in children]
            Q_fitness = [x.evaluate(self.objective_fun) for x in Q]
            self.evals += self.lambdaa
            self.gens += 1

            # Combine population and offspring
            P_parents = P
            R = P + Q
            if self.eliminate_duplicates:
                seen, unique, dups = set(), [], []
                for ind in R:
                    key = self.tree_key(ind.x)
                    (dups if key in seen else unique).append(ind)
                    seen.add(key)
                self.duplicates_removed += max(0, len(dups) - max(0, self.mu - len(unique)))
                R = unique + dups[:max(0, self.mu - len(unique))]
            # print(f"|R_{self.gens}| = {len(R)}")
            # print(len(R))

            # Now do the survival selection
            layers = do_fast_nondominated_sorting(R)
            #print(layers)

            P = []
            i = 0
            while i < len(layers) and len(layers[i]) > 0 and ((len(P) + len(layers[i])) <= self.mu):
                P = P + layers[i] # append the entire layer
                i += 1

            # number of elements that need to be selected for to complete to next population
            if len(P) < self.mu:
                no_missing = self.mu - len(P)

                # now do crowding distance calculation on the i-th layer
                remaining = do_crowding_distance(layers[i], fixed = self.fixed_crowding_distance)
                # print(f"{self.gens}: |P| = {len(P)}, missing = {no_missing}, |remaining| = {len(remaining)}")

                remaining.sort(key = lambda e: e.cdistance, reverse = True)

                P = P + remaining[0:no_missing]

            if self.on_generation is not None:
                self.on_generation(self.gens, P_parents, Q, P)

            if self.log_every is not None:
                if (self.gens % self.log_every) == 0:
                    self.history.append(P[:])

        if self.log_every is not None:
            if (self.gens % self.log_every) == 0:
                self.history.append(P[:])

        self.final_population = P

        return P, self.history

    def write_result(self, outpath, total_time):
        # write trajectory/history
        file = os.path.join(outpath, "trajectory.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow(["gen","c1","c2"])
            for i, pop in enumerate(self.history):
                gen = i * self.log_every
                for ind in pop:
                    csvwriter.writerow([str(gen)] + [str(x) for x in ind.fitness])

        # final approximation
        file = os.path.join(outpath, "pf.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow(["c1","c2"])
            for ind in self.final_population:
                csvwriter.writerow([str(x) for x in ind.fitness])

        # time
        file = os.path.join(outpath, "runtime.csv")
        with open(file, "w", newline = '') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter = ",")
            csvwriter.writerow([str(total_time)])

