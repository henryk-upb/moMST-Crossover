"""Classes for evolutionary operators for multi-objective MST problem variants."""
from abc import abstractmethod
from Graph.Graph import Graph
from Graph.Edge import Edge
from Graph.algorithms.MST import KruskalRelink
from Graph.algorithms.BFS import BFS
from Graph.algorithms.DFS import DFS
from Graph.generators.RandomSpanningTreeGenerator import RandomSpanningTreeGenerator
from EA.utils.RandomWeightSampler import WeightSampler, RandomWeightSampler, EquidistantWeightSampler
from EA.utils.EdgeSampler import EdgeSampler, EdgeSamplerUniform
from EA.utils.EdgeOrderer import EdgeOrderer, EdgeOrdererExact

from collections import deque
from typing import Callable

import random


class EAOperator:
    """Abstract evolutionary operator class."""

    def __init__(self):
        """Empty statitics."""
        self.stats = {}

    def get_stats(self) -> dict:
        """Return a dictionary of statistics colleced with the last call to self.create()."""
        return self.stats

    def set_stats(self, stats: dict) -> None:
        """
        Store set of statstics.

        Args:
            stats (dict(str, any)): a dictionary of values.
        """
        self.stats = stats


class Mutator(EAOperator):
    """Abstract mutator class."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """Create a mutant."""
        pass


class Recombinator(EAOperator):
    """Abstract recombinator/crossover class."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def create(self, tree1: Graph, tree2: Graph) -> Graph:
        """Create a child."""
        pass


class RandomWalkMutator(Mutator):
    """
    Random-Walk mutation operator.

    The operator expects a tree. It adds edges randomly to the tree first. Next,
    it performs a random walk using Broder's algorithm to produce another spanning
    tree.

    Spanning tree constraints: supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        edge_sampler: EdgeSampler | None = None,
        would_violate_constraint = lambda tree, e: False,
        max_iter: int | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise RandomWalkMutator object.

        Args:
            graph (Graph): the source graph.
            sigma (int | None): the maximum number of edges to be randomly added.
            edge_sample (EdgeSampler): edge sampling strategy. Defaut is uniform sampling.
            would_violate_constraint (function(Graph, Edge)): function that expects a
                Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
            max_iter (int): Maximum number of random walk steps. Default is n*n where
                n is the number of nodes of graph.
        """
        super().__init__()

        if edge_sampler is None:
            edge_sampler = EdgeSamplerUniform(graph.get_edges())

        assert isinstance(edge_sampler, EdgeSampler)
        assert graph.is_connected()

        self.sigma = sigma
        self.edge_sampler = edge_sampler
        self.would_violate_constraint = would_violate_constraint

        # Limit the number of random walk runs
        if max_iter is None:
            n = graph.get_n()
            max_iter = n * n
        self.max_iter = max_iter
        self.fixed = fixed

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree by applying the random walk operator.

        Args:
            tree (Graph): the parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree or None if the operation fails (i.e., the
            random walk did not finish in time).
        """
        assert tree.is_connected()

        # number of edges to add
        n_edges = self.sigma if self.fixed else random.randint(1, self.sigma)

        # add new edges
        sampled_edges = self.edge_sampler.sample(n_edges)
        for e in sampled_edges:
            # skip edge if it would violate constraint
            if self.would_violate_constraint(tree, e):
                continue
            tree.add_edge(e.copy(), check_exists = True)

        return RandomSpanningTreeGenerator.sample_broder(tree, max_iter = self.max_iter)


class EdgeExchangeMutator(Mutator):
    """
    Classic edge-exchange operator.

    The operator expects a tree and an edge
    sampling strategy. It samples a random edge, inserts the edge into a copy
    of the spanning tree and re
    """

    def __init__(self, edge_sampler: EdgeSampler, k: int = 1) -> None:
        """
        Initialise EdgeExchangeMutator object.

        Args:
            edge_sampler (Graph.utils.EdgeSampler): edges are sampled from this
                class. If an already existing edge is sampled nothing happens (null operation).
            k (int): number of consecutive edge-exchanges (default is 1).
        """
        super().__init__()

        assert isinstance(edge_sampler, EdgeSampler)
        assert k >= 1

        self.edge_sampler = edge_sampler
        self.k = k

    def get_cycle_edges(self, start: int) -> list[Edge]:
        """Identify edges on cycle."""
        n = self.child.get_n()

        cycle_edges = []

        # keep track of visited nodes in DFS
        visited = [False] * (n + 1)

        # implicitely keep track of DFS tree, i.e., pi[v] = w means {w, v} is a DFS-tree edge
        pi = [0] * (n + 1)

        # weights of DFS-tree edges
        wi = [0] * (n + 1)

        # init search structures
        stack = deque()
        stack.append(start)
        pi[start] = start

        while len(stack) > 0:
            v = stack.pop()

            if visited[v]:
                continue

            visited[v] = True

            # TODO: here we extract the entire neighborhood (need to )
            for e in self.child.adjacency[v].values():
                # get adjacent node
                w = e.other(v)

                # if v was not yet seen, update the tree and put v on stack
                if not visited[w]:
                    pi[w] = v
                    wi[w] = e.get_cost()
                    stack.append(w)

                # found circle -> reconstruct cycle edges and return
                if visited[w] and not (w == pi[v]):
                    cycle_edges.append(e)
                    while not pi[v] == v:
                        u = pi[v]
                        cycle_edges.append(Edge(u, v, [wi[v]]))
                        v = u
                    return cycle_edges

        return cycle_edges

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Create new solution.

        Args:
            tree (Graph): the parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            Object of type EdgeExchangeMutator. Obtain the solution by calling the get_solution()
            method.
        """
        assert tree.is_spanning_tree()

        n = tree.get_n()

        # init copy: Theta(n)
        self.child = tree.copy()

        # sample k new edges
        new_edges = self.edge_sampler.sample(self.k)

        # note: set to zero such that it is well-defined even if no add is added
        self.set_stats({'no_cycle_edges': 0})

        for e in new_edges:

            # add with existance check: O(n)
            added = self.child.add_edge(e, check_exists = True)

            # nothing to do if the edge was not added
            if not added:
                assert self.child.is_spanning_tree()
                continue

            assert self.child.get_m() == n
            assert not self.child.is_spanning_tree()

            # self.child.plot()

            # otherwise delete edge on cycle: O(n)
            edges_on_cycle = self.get_cycle_edges(e.v)

            self.set_stats({'no_cycle_edges': len(edges_on_cycle)})

            # select random edge from identified cycle
            f = edges_on_cycle[random.randint(0, len(edges_on_cycle) - 1)]

            # get rid of a random cycle edge: O(n)
            assert not self.child.is_spanning_tree()
            self.child.delete_edge(f.v, f.w)
            assert self.child.is_spanning_tree()
            # self.child.plot()

        return self.child


class InsertionFirstGlobalMutator(Mutator):
    """
    (Global) Insertion first (IF-G) mutation operator.

    This operator takes a spanning tree,
    adds a couple of random edges, scalarises the weights with a random weight
    vector and applies the single-objectve exact MST algorithm by Kruskal
    to obtain another spanning tree.

    Spanning tree constraints: supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        edge_sampler: EdgeSampler | None = None,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        would_violate_constraint: Callable[[Graph, Edge], bool] = lambda tree, e: False,
        fixed: bool = False
    ) -> None:
        """
        Initialise an InsertionFirstGlobalMutator object.

        Args:
            graph (Graph): the source graph.
            sigma (int | None): the maximum number of edges to be randomly added.
            edge_sampler (EdgeSampler): edge sampling strategy. Defaut is uniform sampling.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            weight_sampler (RandomWeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            would_violate_constraint (function(Graph, Edge)): function that expects a
                Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
            fixed (bool): is the number of randomly added edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert sigma >= 1

        self.sigma = sigma
        self.edge_sampler = edge_sampler if edge_sampler is not None else EdgeSamplerUniform(graph.get_edges())
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
        self.would_violate_constraint = would_violate_constraint
        self.graph = graph
        self.fixed = fixed

        assert isinstance(self.edge_sampler, EdgeSampler)
        assert isinstance(self.weight_sampler, RandomWeightSampler)

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree by applying the (global) insertion-first operator.

        Args:
            tree (Graph): the parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree.
        """
        q = self.graph.get_q()

        self.child = tree.copy()

        # number of edges to add
        n_edges = self.sigma if self.fixed else random.randint(1, self.sigma)

        self.set_stats({'no_added_edges': n_edges})

        # add new edges
        sampled_edges = self.edge_sampler.sample(n_edges)

        for e in sampled_edges:
            # skip edge if it would violate the maximum degree constraint
            if self.would_violate_constraint(self.child, e):
                continue
            self.child.add_edge(e.copy(), check_exists = False)

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        # extract relevant edges and pre-process
        relevant_edges = self.edge_orderer.order(self.child.get_edges(), lambdas)

        # generate final spanning tree based on weight vector
        self.child = KruskalRelink(self.child, relevant_edges).solve().get_solution()
        # self.child.plot()
        return self.child


class InsertionFirstLocalMutator(Mutator):
    """
    (Local) Insertion first (IF-L) mutation operator.

    This operator takes a spanning tree,
    adds a couple of random edges between a subset of nodes , scalarises the weights
    with a random weight vector and applies the single-objectve exact MST algorithm by Kruskal
    to obtain another spanning tree.

    Spanning tree constraints: supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma_nodes: int | None,
        sigma_edges: int | None,
        traversal_method: str = 'bfs',
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        would_violate_constraint: Callable[[Graph, Edge], bool] = lambda tree, e: False,
        fixed: bool = False
    ) -> None:
        """
        Initialise an InsertionFirstLocalMutator object.

        Args:
            graph (Graph): the source graph.
            sigma_nodes (int | None): the maximum depth of the nodes to select.
            sigma_edges (int | None): the maximum number of edges to be randomly added.
            traversal_method (str): Either 'bfs' or 'dfs' for {breadth,depth} first
                search respectively. This decides on how the sub-graph is build.
                Default is 'bfs'.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            weight_sampler (RandomWeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            would_violate_constraint (function(Graph, Edge)): function that expects a
            Graph (tree) and an Edge object e and returns a Boolean. A return value
                True means that adding e to tree would violate a constraint on the tree.
                Default is no constraint, i.e., a function that always returns False.
            fixed (bool): is the number of randomly added edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert 3 <= sigma_nodes <= graph.get_n()
        assert sigma_edges >= 1

        self.sigma_nodes = sigma_nodes
        self.sigma_edges = sigma_edges
        self.traversal_method = BFS if traversal_method == 'bfs' else DFS
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
        self.would_violate_constraint = would_violate_constraint
        self.graph = graph
        self.fixed = fixed

        assert isinstance(self.weight_sampler, RandomWeightSampler)

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree by applying the (local) insertion-first operator.

        Args:
            tree (Graph): the parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree.
        """
        n = self.graph.get_n()
        q = self.graph.get_q()

        self.child = tree.copy()

        # sample start node for limited BFS call
        start_node = random.randint(1, n)

        # obtain selected nodes (these are used for subgraph construction)
        selected_nodes = self.traversal_method(
            tree, start = start_node, max_visited = self.sigma_nodes
        ).get_queued_nodes()

        # number of edges to add
        n_edges = self.sigma_edges if self.fixed else random.randint(1, self.sigma_edges)
        edge_sampler = EdgeSamplerUniform(self.graph.get_edges(selected_nodes))

        self.set_stats({'no_selected_nodes': len(selected_nodes), 'no_added_edges': n_edges})

        # add new edges
        sampled_edges = edge_sampler.sample(n_edges)
        for e in sampled_edges:
            # skip edge if it would violate the maximum degree constraint
            if self.would_violate_constraint(self.child, e):
                continue
            self.child.add_edge(e.copy(), check_exists = False)

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        # extract relevant edges and pre-process
        relevant_edges = self.edge_orderer.order(self.child.get_edges(), lambdas)

        # generate final spanning tree based on weight vector
        self.child = KruskalRelink(self.child, relevant_edges).solve().get_solution()
        # self.child.plot()
        return self.child


class USGMutator(Mutator):
    """
    Unconnected sub-graph mutation operator.

    This operator takes a spanning tree, drops a couple of random edges, scalarises
    the weights of the source graph and reconnects the connected components
    by applying Kruskal.

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        filter_edges: bool = False,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise a DeletionFirstUSGMutator object.

        Args:
            graph (Graph): the source graph.
            sigma (int|None): max. value for the number of edges to be randomly dropped.
            filter_edges (bool): should edges be preprocessed in the sense that
                edges that would certainly close a cycle are being ignored before
                sorting takes place? Defaults to False.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            weight_sampler (RandomWeightSampler): an object of class RandomWeightSampler
            fixed (bool): is the number of randomly dropped edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert isinstance(graph, Graph)
        assert 1 <= sigma <= (graph.get_n() - 1)

        self.graph = graph
        self.sigma = sigma
        self.filter_edges = filter_edges
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
        self.fixed = fixed
        self.relevant_edges = self.graph.get_edges()

    def _create_multiple(
        self,
        initial_tree: Graph,
        n_children: int = 1
    ) -> list[Graph]:
        """
        Create list of offspring spanning trees based on Kruskal-Relinking.

        Note: private utility method.

        Args:
            graph (Graph): graph that is passed to KruskalRelink.
            relevant_edges (list[Edge]): edges used for relinking.
            initial_tree (Graph): initial tree after USGS dropped edges.
            n_children (int): the number of children to be generated.

        Returns:
            list[Graph]: a list of spanning trees.
        """
        # sample weight vectors
        weight_vectors = self.weight_sampler.sample(n = initial_tree.get_q(), k = n_children)

        # use the weight vectors to bring

        return [
            KruskalRelink(
                graph = self.graph,
                edges = self.edge_orderer.order(self.relevant_edges, lambdas),
                initial_tree = initial_tree
            ).solve().get_solution()
            for lambdas in weight_vectors
        ]

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree by applying the deletion-first USGMutator operator.

        Args:
            tree (Graph): the parent spanning tree.
            n_children (int): number of children to produce. This makes most sense
                'filter_edges = True'. Default is 1.

        Returns:
            Graph | list[Graph]: a spanning tree or a list of spanning trees if
                'n_children' is larger than 1.
        """
        # number of edges to delete
        n_edges = self.sigma if self.fixed else random.randint(1, self.sigma)

        self.set_stats({'no_dropped_edges': n_edges})

        # create forest of remaining subtrees after removal of edges
        initial_tree = Graph.get_subgraph_by_deleting_random_edges(tree, n_edges)

        # filter relevant edges
        if self.filter_edges:
            self.relevant_edges = self.graph.get_relinking_edges(initial_tree)

        offspring = self._create_multiple(
            initial_tree = initial_tree,
            n_children = n_children
        )

        # Should all operators just return a list?
        if n_children == 1:
            return offspring[0]


class USGMutatorWithPresorting(Mutator):
    """
    Unconnected sub-graph mutation operator adopting pre-sorted edge sets.

    This operator takes a spanning tree, drops a couple of random edges, and
    reconnects the connected components using pre-sorted edge lists.
    This saves a factor of order O(log(m)) = O(log(n^2)) = O(log(n))

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        L: int | None = None,
        edge_sets: list[list[Edge]] | None = None,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: RandomWeightSampler | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise a USGSMutatorWithPresorting object.

        Args:
            graph (Graph): the source graph.
            sigma (int|None): max. value for the number of edges to be randomly dropped.
            edge_sets (list[list[Edge]]|None): List of lists of pre-sorted edges.
            L (int|None): number of weight vectors to sample and do pre-sorting.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            weight_sampler (RandomWeightSampler): an object of class RandomWeightSampler.
                Default is EquidistantWeightSampler.
            fixed (bool): is the number of randomly dropped edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert isinstance(graph, Graph)
        assert 1 <= sigma <= (graph.get_n() - 1)

        self.graph = graph
        self.sigma = sigma
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else EquidistantWeightSampler()
        self.edge_sets = edge_sets if edge_sets is not None else self._pre_sort_edges(graph, L)
        self.L = L if L is not None else len(self.edge_sets)
        self.fixed = fixed


    def _pre_sort_edges(self, graph: Graph, L: int) -> list[Edge]:
        edges = graph.get_edges()
        weight_vectors = self.weight_sampler.sample(graph.get_q(), L)
        return [self.edge_orderer.order(edges, weight) for weight in weight_vectors]


    def create(self, tree: Graph) -> Graph | list[Graph]:
        """
        Produce a new spanning tree by applying the USGMutator with pre-sorted edges.

        Args:
            tree (Graph): the parent spanning tree.

        Returns:
            Graph: a spanning tree.
        """
        # number of edges to delete
        n_edges = self.sigma if self.fixed else random.randint(1, self.sigma)

        self.set_stats({'no_dropped_edges': n_edges})

        # create forest of remaining subtrees after removal of edges
        initial_tree = Graph.get_subgraph_by_deleting_random_edges(tree, n_edges)

        # sample pre-sorted edge set at random
        i = random.randint(0, self.L - 1)
        sorted_edges = self.edge_sets[i]
        sorted_relevant_edges = self.graph.get_relinking_edges(initial_tree, sorted_edges)

        return KruskalRelink(
            graph = self.graph,
            edges = sorted_relevant_edges,
            initial_tree = initial_tree
            ).solve().get_solution()


class UnionCrossover(Recombinator):
    

    def __init__(
        self,
        graph: Graph,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
    
    def create(self, tree1: Graph, tree2: Graph) -> Graph:
        """
        Produce a new spanning tree based on the UnionCrossover logic.

        Args:
            tree1 (Graph): the first parent spanning tree.
            tree2 (Graph): the first parent spanning tree.

        Returns:
            A spanning tree.
        """
        assert tree1.is_spanning_tree()
        assert tree2.is_spanning_tree()

        edgeList = tree1.copy()
        for e in tree2.get_edges():
            edgeList.add_edge(e.copy(), check_exists=True)
        q = tree1.get_q()
        lambdas = self.weight_sampler.sample(q)[0]

        relevant_edges = self.edge_orderer.order(edgeList.get_edges(), lambdas)

        self.child = KruskalRelink(
            self.graph,
            relevant_edges,
        ).solve().get_solution()

        return self.child
    
class IntersectionCrossover(Recombinator):
    """
    Intersection Crossover (IX) recombination operator.

    Edges shared by both parent spanning trees (the intersection) are inherited
    directly into the child and stay fixed. The remaining edges needed to
    complete the spanning tree are chosen by Kruskal-relinking over a candidate
    pool made up of the union of both parents' edges plus a linear (in the
    number of nodes) number of additional randomly sampled edges from the
    source graph, scalarised with a random weight vector.

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        edge_sampler: EdgeSampler | None = None,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        sigma: int | None = None,
    ) -> None:
        """
        Initialise an IntersectionCrossover object.

        Args:
            graph (Graph): the source graph.
            edge_sampler (EdgeSampler): sampling strategy for the additional
                random edges. Defaults to uniform sampling over all edges of
                'graph'.
            edge_orderer (EdgeOrderer): strategy for ordering the candidate edges
                before passing down to KruskalRelink. Defaults to an instance of
                EdgeOrdererExact.
            weight_sampler (WeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            sigma (int | None): number of additional random edges to sample into
                the candidate pool. Defaults to graph.get_n() (i.e., linear in the
                number of nodes).
        """
        super().__init__()

        self.graph = graph
        self.edge_sampler = edge_sampler if edge_sampler is not None else EdgeSamplerUniform(graph.get_edges())
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
        self.sigma = sigma if sigma is not None else graph.get_n()

    def create(self, tree1: Graph, tree2: Graph) -> Graph:
        """
        Produce a new spanning tree based on the IntersectionCrossover logic.

        Args:
            tree1 (Graph): the first parent spanning tree.
            tree2 (Graph): the second parent spanning tree.

        Returns:
            A spanning tree.
        """
        assert tree1.is_spanning_tree()
        assert tree2.is_spanning_tree()

        q = tree1.get_q()

        # child starts out empty; edges present in both parents (the
        # intersection) are inherited directly and stay fixed from here on
        self.child = tree1.copy(copy_edges = False)

        # union of both parents' edges: start with tree1's edges and add
        # tree2's edges that are not already part of tree1
        union_edges = list(tree1.get_edges())

        for e in tree2.get_edges():
            if tree1.has_edge(e.v, e.w):
                self.child.add_edge(e.copy(), check_exists = False)
            else:
                union_edges.append(e)

        # candidate pool for relinking: the union of both parents plus a
        # linear number of additional randomly sampled edges
        sampled_edges = self.edge_sampler.sample(self.sigma)
        candidate_edges = union_edges + sampled_edges

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        relevant_edges = self.edge_orderer.order(candidate_edges, lambdas)

        self.child = KruskalRelink(
            self.graph,
            relevant_edges,
            self.child
        ).solve().get_solution()

        assert self.child.is_spanning_tree()

        return self.child
    
class XORCrossover(Recombinator):
    """
    XOR Crossover (XOR) recombination operator.

    Edges shared by both parent spanning trees (the intersection) are inherited
    directly into the child and stay fixed. The remaining edges needed to
    complete the spanning tree are chosen by Kruskal-relinking over a candidate
    pool made up of the symmetric difference of both parents' edges plus a linear
    (in the number of nodes) number of additional randomly sampled edges from the
    source graph, scalarised with a random weight vector.

    Spanning tree constraints: not supported.
    """
    
    def __init__(
        self,
        graph: Graph,
        edge_sampler: EdgeSampler | None = None,
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        sigma: int | None = None,
    ) -> None:
        """
        Initialise an XORCrossover object.

        Args:
            graph (Graph): the source graph.
            edge_sampler (EdgeSampler): sampling strategy for the additional
                random edges. Defaults to uniform sampling over all edges of
                'graph'.
            edge_orderer (EdgeOrderer): strategy for ordering the candidate edges
                before passing down to KruskalRelink. Defaults to an instance of
                EdgeOrdererExact.
            weight_sampler (WeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            sigma (int | None): number of additional random edges to sample into
                the candidate pool. Defaults to graph.get_n() (i.e., linear in the
                number of nodes).
        """
        super().__init__()

        self.graph = graph
        self.edge_sampler = edge_sampler if edge_sampler is not None else EdgeSamplerUniform(graph.get_edges())
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()
        self.sigma = sigma if sigma is not None else graph.get_n()
    
    def create(self, tree1: Graph, tree2: Graph) -> Graph:
        """
        Produce a new spanning tree based on the XORCrossover logic.

        Args:
            tree1 (Graph): the first parent spanning tree.
            tree2 (Graph): the second parent spanning tree.

        Returns:
            A spanning tree.
        """
        assert tree1.is_spanning_tree()
        assert tree2.is_spanning_tree()

        q = tree1.get_q()

        # child starts out empty; edges present in both parents (the
        # intersection) are inherited directly and stay fixed from here on
        self.child = tree1.copy(copy_edges = False)

        # symmetric difference of both parents' edges: edges present in
        # exactly one of the two parent trees
        sym_diff_edges = [e for e in tree1.get_edges() if not tree2.has_edge(e.v, e.w)]

        for e in tree2.get_edges():
            if tree1.has_edge(e.v, e.w):
                self.child.add_edge(e.copy(), check_exists = False)
            else:
                sym_diff_edges.append(e)

        # candidate pool for relinking: the symmetric difference of both
        # parents plus a linear number of additional randomly sampled edges
        sampled_edges = self.edge_sampler.sample(self.sigma)
        candidate_edges = sym_diff_edges + sampled_edges

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        relevant_edges = self.edge_orderer.order(candidate_edges, lambdas)

        self.child = KruskalRelink(
            self.graph,
            relevant_edges,
            self.child
        ).solve().get_solution()

        assert self.child.is_spanning_tree()

        return self.child

class SGSECJMutator(Mutator):
    """
    Local version of the scalarised sub-graph mutation operator (SGS) proposed in [1, 2].

    Note: this is the re-implementation of the operator as it was used for experimentation
    in [2].

    The algorithm extracts an induced sub-graph of the input tree adopting limited BFS. It next calculates
    an MST of the sub-graph by first reducing the edge weights to scalar values by a weighted-sum approach
    and applying Kruskal's single-objective MST

    [1] J. Bossek and C. Grimme, ‘A pareto-beneficial sub-tree mutation for the multi-criteria minimum spanning tree
    problem’, in 2017 IEEE Symposium Series on Computational Intelligence (SSCI), Nov. 2017, pp. 1–8.
    doi: 10.1109/SSCI.2017.8285183.

    [2] J. Bossek and C. Grimme, ‘On single-objective sub-graph-based mutation for solving the bi-objective minimum
    spanning tree problem’, Evolutionary Computation, 2023. (under review)

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        traversal_method: str = 'bfs',
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise an SG object.

        Args:
            graph (Graph): Input graph.
            sigma (int): Maximum number of nodes for the limited BFS call. Must
                be between 3 and |V|. In every application of the operator a number
                in {3, ..., sigma} is sampled at random for the sub-graph size.
            traversal_method (str): Either 'bfs' or 'dfs' for {breadth,depth} first
                search respectively. This decides on how the sub-graph is build.
                Default is 'bfs'.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            sampler (RandomWeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            fixed (bool): is the number of randomly added edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert isinstance(graph, Graph)
        assert 3 <= sigma <= graph.get_n()
        assert traversal_method in ['bfs', 'dfs']

        self.graph = graph
        self.sigma = sigma
        self.traversal_method = BFS if traversal_method == 'bfs' else DFS
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()

        # book-keeping of visited nodes
        self.visited = [False] * (graph.get_n() + 1)
        self.fixed = fixed

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree based on the SGS operator logic.

        Args:
            tree (Graph): parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree.
        """
        n = tree.get_n()
        q = tree.get_q()

        # sample start node for limited BFS call
        start_node = random.randint(1, n)

        # override if parameter is passed to this method
        sigma = self.sigma if self.sigma is not None else int(self.graph.get_n() / 2)

        # number of nodes for limited BFS
        max_visited = sigma if self.fixed else random.randint(3, self.sigma)

        # obtain selected nodes (these are used for subgraph construction)
        selected_nodes = self.traversal_method(
            tree, start = start_node, max_visited = max_visited
        ).get_queued_nodes()

        self.set_stats({'no_selected_nodes': max_visited})

        # temporarily set the visited flag for all selected nodes
        for v in selected_nodes:
            self.visited[v] = True

        assert sum(self.visited) == len(selected_nodes)

        # make a copy of the source tree
        self.child = tree.copy(copy_edges = True)

        # now remove edges from the copy. This takes O(s n) and not O(s log n)
        # as claimed since we use list and not balanced trees
        for e in self.child.get_edges():
            if self.visited[e.v] and self.visited[e.w]:
                self.child.delete_edge(e.v, e.w)

        # get induced sub-graph
        induced_graph = Graph.induced(self.graph, selected_nodes)
        # induced_graph.plot()

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        # calculate MST on induced sub-graph and extract its edges
        relevant_edges = self.edge_orderer.order(induced_graph.get_edges(), lambdas)
        induced_graph_mst = KruskalRelink(
            induced_graph, relevant_edges
        ).solve().get_solution()
        # G_induced_mst.plot()

        new_edges = induced_graph_mst.get_edges()
        # T.plot()
        # self.child.plot()

        # finally copy all edges from the MST on the sub-graph
        # and map their node IDs (1, ..., max_visited) to the original set
        # selected_nodes
        for e in new_edges:
            v_induced, w_induced, cost = e.v, e.w, e.get_cost()
            # map back to original node Ids
            v, w = selected_nodes[v_induced - 1], selected_nodes[w_induced - 1]
            self.child.add_edge(Edge(v, w, cost), check_exists = False)

        # self.child.plot()

        assert self.child.is_spanning_tree()

        # reset visited flag in O(|selected_nodes|)
        for v in selected_nodes:
            self.visited[v] = False

        assert sum(self.visited) == 0

        return self.child


class SGSImprovedMutator(Mutator):
    """
    Improved ECJ-version of the local version of the scalarised sub-graph mutation operator (SGS) proposed in [1, 2].

    Note: the improvement is that edges are no longer 'actively' deleted from the
    copy of the spanning tree. Instead a new forest is created.

    The algorithm extracts an induced sub-graph of the input tree adopting limited BFS. It next calculates
    an MST of the sub-graph by first reducing the edge weights to scalar values by a weighted-sum approach
    and applying Kruskal's single-objective MST

    [1] J. Bossek and C. Grimme, ‘A pareto-beneficial sub-tree mutation for the multi-criteria minimum spanning tree
    problem’, in 2017 IEEE Symposium Series on Computational Intelligence (SSCI), Nov. 2017, pp. 1–8.
    doi: 10.1109/SSCI.2017.8285183.

    [2] J. Bossek and C. Grimme, ‘On single-objective sub-graph-based mutation for solving the bi-objective minimum
    spanning tree problem’, Evolutionary Computation, 2023. (under review)

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        traversal_method: str = 'bfs',
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise an SG object.

        Args:
            graph (Graph): Input graph.
            sigma (int): Maximum number of nodes for the limited BFS call. Must
                be between 3 and |V|. In every application of the operator a number
                in {3, ..., sigma} is sampled at random for the sub-graph size.
            traversal_method (str): Either 'bfs' or 'dfs' for {breadth,depth} first
                search respectively. This decides on how the sub-graph is build.
                Default is 'bfs'.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            sampler (RandomWeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            fixed (bool): is the number of randomly added edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert isinstance(graph, Graph)
        assert 3 <= sigma <= graph.get_n()
        assert traversal_method in ['bfs', 'dfs']

        self.graph = graph
        self.sigma = sigma
        self.traversal_method = BFS if traversal_method == 'bfs' else DFS
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()

        # book-keeping of visited nodes
        self.visited = [False] * (graph.get_n() + 1)
        self.fixed = fixed

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree based on the SGS operator logic.

        Args:
            tree (Graph): parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree.
        """
        n = tree.get_n()
        q = tree.get_q()

        # sample start node for limited BFS call
        start_node = random.randint(1, n)

        # override if parameter is passed to this method
        sigma = self.sigma if self.sigma is not None else int(self.graph.get_n() / 2)

        # number of nodes for limited BFS
        max_visited = sigma if self.fixed else random.randint(3, self.sigma)

        # obtain selected nodes (these are used for subgraph construction)
        selected_nodes = self.traversal_method(
            tree, start = start_node, max_visited = max_visited
        ).get_queued_nodes()

        self.set_stats({'no_selected_nodes': max_visited})

        # temporarily set the visited flag for all selected nodes
        for v in selected_nodes:
            self.visited[v] = True

        assert sum(self.visited) == len(selected_nodes)

        # get induced sub-graph
        induced_graph = Graph.induced(self.graph, selected_nodes)
        # induced_graph.plot()

        # sample random weight vector
        lambdas = self.weight_sampler.sample(q)[0]

        # calculate MST on induced sub-graph and extract its edges
        relevant_edges = self.edge_orderer.order(induced_graph.get_edges(), lambdas)
        induced_graph_mst = KruskalRelink(
            induced_graph, relevant_edges
        ).solve().get_solution()
        # G_induced_mst.plot()

        new_edges = induced_graph_mst.get_edges()
        parent_edges = tree.get_edges()
        # T.plot()

        # prepare empty tree
        self.child = tree.copy(copy_edges = False)

        # self.child.plot()

        # now copy all edges from parent tree where at least one end point is not
        # in the induced sub-graph
        for e in parent_edges:
            if (not self.visited[e.v]) or not self.visited[e.w]:
                self.child.add_edge(e.copy(), check_exists = False)

        # self.child.plot()

        # finally copy all edges from the MST on the sub-graph
        # and map their node IDs (1, ..., max_visited) to the original set
        # selected_nodes
        for e in new_edges:
            v_induced, w_induced, cost = e.v, e.w, e.get_cost()
            # map back to original node Ids
            v, w = selected_nodes[v_induced - 1], selected_nodes[w_induced - 1]
            self.child.add_edge(Edge(v, w, cost), check_exists = False)

        # self.child.plot()

        assert self.child.is_spanning_tree()

        # reset visited flag in O(|selected_nodes|)
        for v in selected_nodes:
            self.visited[v] = False

        assert sum(self.visited) == 0

        return self.child


class SGSSimplifiedMutator(Mutator):
    """
    SIMPLIFIED local version of the scalarised sub-graph mutation operator (SGS) proposed in [1, 2].

    The algorithm extracts an induced sub-graph of the input tree adopting limited BFS. It next calculates
    an MST of the sub-graph by first reducing the edge weights to scalar values by a weighted-sum approach
    and applying Kruskal's single-objective MST

    [1] J. Bossek and C. Grimme, ‘A pareto-beneficial sub-tree mutation for the multi-criteria minimum spanning tree
    problem’, in 2017 IEEE Symposium Series on Computational Intelligence (SSCI), Nov. 2017, pp. 1–8.
    doi: 10.1109/SSCI.2017.8285183.

    [2] J. Bossek and C. Grimme, ‘On single-objective sub-graph-based mutation for solving the bi-objective minimum
    spanning tree problem’, Evolutionary Computation, 2023. (under review)

    Spanning tree constraints: not supported.
    """

    def __init__(
        self,
        graph: Graph,
        sigma: int | None,
        traversal_method: str = 'bfs',
        edge_orderer: EdgeOrderer | None = None,
        weight_sampler: WeightSampler | None = None,
        fixed: bool = False
    ) -> None:
        """
        Initialise an SG object.

        Args:
            graph (Graph): Input graph.
            sigma (int): Maximum number of nodes for the limited BFS call. Must
                be between 3 and |V|. In every application of the operator a number
                in {3, ..., sigma} is sampled at random for the sub-graph size.
            traversal_method (str): Either 'bfs' or 'dfs' for {breadth,depth} first
                search respectively. This decides on how the sub-graph is build.
                Default is 'bfs'.
            edge_orderer (EdgeOrderer): strategy for ordering the edges before passing down to
                KruskalRelink. Defaults to an instance of EdgeOrdererExact.
            sampler (RandomWeightSampler): an object of class RandomWeightSampler
                which is used to sample weights for the scalarisation part.
            fixed (bool): is the number of randomly added edges an upper bound or fixed?
                Default is False.
        """
        super().__init__()

        assert isinstance(graph, Graph)
        assert 3 <= sigma <= graph.get_n()
        assert traversal_method in ['bfs', 'dfs']

        self.graph = graph
        self.sigma = sigma
        self.traversal_method = BFS if traversal_method == 'bfs' else DFS
        self.edge_orderer = edge_orderer if edge_orderer is not None else EdgeOrdererExact()
        self.weight_sampler = weight_sampler if weight_sampler is not None else RandomWeightSampler()

        # book-keeping of visited nodes
        self.visited = [False] * (graph.get_n() + 1)
        self.fixed = fixed

    def create(self, tree: Graph, n_children: int = 1) -> Graph | list[Graph]:
        """
        Produce a new spanning tree based on the SGS operator logic.

        Args:
            tree (Graph): parent spanning tree.
            n_children (int): number of children. Values other than the default 1
                are not supported by this operator.

        Returns:
            A spanning tree.
        """
        n = tree.get_n()

        # sample start node for limited search call
        start_node = random.randint(1, n)

        # override if parameter is passed to this method
        sigma = self.sigma if self.sigma is not None else int(self.graph.get_n() / 2)

        # number of nodes for limited search
        max_visited = sigma if self.fixed else random.randint(3, sigma)

        self.set_stats({'no_selected_nodes': max_visited})

        # obtain selected nodes (these are used for subgraph construction)
        selected_nodes = self.traversal_method(
            tree, start = start_node, max_visited = max_visited
        ).get_queued_nodes()

        # temporarily set the visited flag for all selected nodes
        for v in selected_nodes:
            self.visited[v] = True
        assert sum(self.visited) == len(selected_nodes)

        # copy of parent tree
        initial_tree = tree.copy(copy_edges = False)
        # copy only edges where at most one incident node was selected
        for e in tree:
            if (not self.visited[e.v]) or (not self.visited[e.w]):
                initial_tree.add_edge(e.copy(), check_exists = False)

        # now prepare the "relevant" edges
        relevant_edges = [
            e for e in self.graph.get_edges(selected_nodes)
            if self.visited[e.v] and self.visited[e.w]
        ]

        # sample random weight vector
        lambdas = self.weight_sampler.sample(tree.get_q())[0]

        relevant_edges = self.edge_orderer.order(relevant_edges, lambdas)

        self.child = KruskalRelink(
            self.graph, relevant_edges, initial_tree
        ).solve().get_solution()
        # self.child.plot()

        assert self.child.is_spanning_tree()

        # reset visited flag in O(|selected_nodes|)
        for v in selected_nodes:
            self.visited[v] = False

        assert sum(self.visited) == 0

        return self.child
