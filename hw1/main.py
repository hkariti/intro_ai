from framework import *
from deliveries import *

from matplotlib import pyplot as plt
import numpy as np
from typing import List, Union

# Load the map
roads = load_map_from_csv(Consts.get_data_file_path("tlv.csv"))

# Make `np.random` behave deterministic.
Consts.set_seed()


# --------------------------------------------------------------------
# -------------------------- Map Problem -----------------------------
# --------------------------------------------------------------------

def plot_distance_and_expanded_wrt_weight_figure(
        weights: Union[np.ndarray, List[float]],
        total_distance: Union[np.ndarray, List[float]],
        total_expanded: Union[np.ndarray, List[int]]):
    """
    Use `matplotlib` to generate a figure of the distance & #expanded-nodes
     w.r.t. the weight.
    """
    assert len(weights) == len(total_distance) == len(total_expanded)

    fig, ax1 = plt.subplots()

    ax1.set_title('Cost and quality of solution wrt heuristic weight')
    ax1.plot(weights, total_distance, 'b-')

    # ax1: Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('distance traveled', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_xlabel('weight')

    # Create another axis for the #expanded curve.
    ax2 = ax1.twinx()

    ax2.plot(weights, total_expanded, 'r-')

    # ax2: Make the y-axis label, ticks and tick labels match the line color.
    ax2.set_ylabel('states expanded', color='r')
    ax2.tick_params('y', colors='r', right=True)

    fig.tight_layout()
    plt.show()


def run_astar_for_weights_in_range(heuristic_type: HeuristicFunctionType, problem: GraphProblem):
    weights = np.linspace(0.5, 1, 20)
    costs = []
    expanded_states = []
    for w in weights:
        astar = AStar(heuristic_type, heuristic_weight=w)
        res = astar.solve_problem(problem)
        costs.append(res.final_search_node.cost)
        expanded_states.append(res.nr_expanded_states)

    plot_distance_and_expanded_wrt_weight_figure(weights, costs, expanded_states)


def map_problem():
    print()
    print('Solve the map problem.')

    # Ex.8
    map_prob = MapProblem(roads, 54, 549)
    uc = UniformCost()
    res = uc.solve_problem(map_prob)
    print(res)

    # Ex.10
    astar = AStar(NullHeuristic)
    res = astar.solve_problem(map_prob)
    print(res)

    # Ex.11
    astar = AStar(AirDistHeuristic)
    res = astar.solve_problem(map_prob)
    print(res)

    # Ex.12
    run_astar_for_weights_in_range(AirDistHeuristic, map_prob)


# --------------------------------------------------------------------
# ----------------------- Deliveries Problem -------------------------
# --------------------------------------------------------------------

def relaxed_deliveries_problem():

    print()
    print('Solve the relaxed deliveries problem.')

    big_delivery = DeliveriesProblemInput.load_from_file('big_delivery.in', roads)
    big_deliveries_prob = RelaxedDeliveriesProblem(big_delivery)

    # Ex.16
    astar = AStar(MaxAirDistHeuristic)
    res = astar.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.17
    astar = AStar(MSTAirDistHeuristic)
    res = astar.solve_problem(big_deliveries_prob)
    print(res)

    # Ex.18
    run_astar_for_weights_in_range(MSTAirDistHeuristic, big_deliveries_prob)

    # Ex.24
    greedy_bf = AStar(MSTAirDistHeuristic, heuristic_weight=1)
    astar = AStar(MSTAirDistHeuristic)
    # Get cost per iteration for Greedy Stochastic and Anytime Greedy Stochastic
    iterations = 100
    cost_per_iteration = []
    best_cost_per_iteration = []
    best_cost = np.inf
    for i in range(iterations):
        greedy_stochastic = GreedyStochastic(MSTAirDistHeuristic)
        res = greedy_stochastic.solve_problem(big_deliveries_prob)
        current_cost = res.final_search_node.cost
        cost_per_iteration.append(current_cost)
        best_cost = np.min([best_cost, current_cost])
        best_cost_per_iteration.append(best_cost)
    # Get cost for AStar and Greedy Best First 
    astar_cost = astar.solve_problem(big_deliveries_prob).final_search_node.cost
    greedy_bf_cost = greedy_bf.solve_problem(big_deliveries_prob).final_search_node.cost
    # Plot results
    fig, ax1 = plt.subplots()
    ax1.plot(range(iterations), cost_per_iteration, 'b-')
    ax1.plot(range(iterations), best_cost_per_iteration, 'g-')
    ax1.axhline(astar_cost, color='r')
    ax1.axhline(greedy_bf_cost, color='k')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('States expanded')
    ax1.legend(['GreedyStochastic', 'AnytimeGreedyStochastic', 'A*', 'GreedyBestFirst'])
    ax1.set_title('Cost wrt #Iterations for different algorithms')
    plt.show()


def strict_deliveries_problem():
    print()
    print('Solve the strict deliveries problem.')

    small_delivery = DeliveriesProblemInput.load_from_file('small_delivery.in', roads)
    small_deliveries_strict_problem = StrictDeliveriesProblem(
        small_delivery, roads, inner_problem_solver=AStar(AirDistHeuristic))

    # Ex.26
    run_astar_for_weights_in_range(MSTAirDistHeuristic, small_deliveries_strict_problem)

    # Ex.28
    astar = AStar(RelaxedDeliveriesHeuristic)
    res = astar.solve_problem(small_deliveries_strict_problem)
    print(res)


def main():
    map_problem()
    relaxed_deliveries_problem()
    strict_deliveries_problem()


if __name__ == '__main__':
    main()
