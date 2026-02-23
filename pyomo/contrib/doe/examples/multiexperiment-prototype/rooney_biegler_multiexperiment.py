#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
"""
Test script for multi-experiment optimization using the Rooney-Biegler example.
This script demonstrates the use of optimize_experiments() method.

Based on Rooney, W. C. and Biegler, L. T. (2001). Design for model parameter
uncertainty using nonlinear confidence regions. AIChE Journal, 47(8), 1794-1804.
"""

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.common.dependencies import pandas as pd, numpy as np
import json
from pathlib import Path


def rooney_biegler_model(data, theta=None):
    model = pyo.ConcreteModel()

    if theta is None:
        theta = {'asymptote': 15, 'rate_constant': 0.5}

    model.asymptote = pyo.Var(initialize=theta['asymptote'])
    model.rate_constant = pyo.Var(initialize=theta['rate_constant'])

    # Fix the unknown parameters
    model.asymptote.fix()
    model.rate_constant.fix()

    # Add the experiment inputs
    model.hour = pyo.Var(initialize=data["hour"].iloc[0], bounds=(0, 10))

    # Fix the experiment inputs
    model.hour.fix()

    # Add the response variable
    model.y = pyo.Var(within=pyo.PositiveReals, initialize=data["y"].iloc[0])

    def response_rule(m):
        return m.y == m.asymptote * (1 - pyo.exp(-m.rate_constant * m.hour))

    model.response_function = pyo.Constraint(rule=response_rule)

    return model


class RooneyBieglerExperiment(Experiment):

    def __init__(self, data, measure_error=None, theta=None):
        self.data = data
        self.model = None
        self.measure_error = measure_error
        self.theta = theta

    def create_model(self):
        # rooney_biegler_model expects a dataframe
        data_df = self.data.to_frame().transpose()
        self.model = rooney_biegler_model(data_df, theta=self.theta)

    def label_model(self):

        m = self.model

        # Add experiment outputs as a suffix
        # Experiment outputs suffix is required for parmest
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_outputs.update([(m.y, self.data['y'])])

        # Add unknown parameters as a suffix
        # Unknown parameters suffix is required for both Pyomo.DoE and parmest
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.asymptote, m.rate_constant]
        )

        # Add measurement error as a suffix
        # Measurement error suffix is required for Pyomo.DoE and
        #  `cov` estimation in parmest
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error.update([(m.y, self.measure_error)])

        # Add hour as an experiment input
        # Experiment inputs suffix is required for Pyomo.DoE
        m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.experiment_inputs.update([(m.hour, self.data['hour'])])

        # For multiple experiments, we need to add symmetry breaking constraints
        # to avoid identical models as a suffix `sym_break_cons`
        m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.sym_break_cons[m.hour] = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()
        return self.model


def run_rooney_biegler_multi_experiment_doe(
    experiment_list, objective_option="determinant", prior_FIM=None, tee=False
):
    """
    Test multi-experiment optimization with the Rooney-Biegler example.

    Parameters
    ----------
    experiment_list : list
        List of RooneyBieglerExperiment objects to optimize simultaneously
    objective_option : str, optional
        Objective function option ('determinant', 'trace', or 'pseudo_trace'), by default 'determinant'
    prior_FIM : np.ndarray, optional
        Prior Fisher Information Matrix, by default None
    tee : bool, optional
        Whether to show solver output, by default False

    Returns
    -------
    DesignOfExperiments
        The DoE object containing optimization results
    """
    # Get number of experiments
    n_exp = len(experiment_list)

    # Create the DesignOfExperiments object
    print(f"Objective: {objective_option}")
    print(f"\n{'='*60}")
    print(f"Testing Multi-Experiment Optimization with {n_exp} experiments")
    print(f"{'='*60}\n")

    # Note: Not using prior_FIM to avoid numerical issues with this simple model
    # Also not passing a custom solver - let DoE use its default
    doe_obj = DesignOfExperiments(
        experiment_list=experiment_list,
        objective_option=objective_option,
        prior_FIM=prior_FIM,
        tee=tee,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    # Run multi-experiment optimization
    print(f"Running optimize_experiments with {n_exp} experiments...")
    doe_obj.optimize_experiments()

    # Print results
    print(f"\n{'='*60}")
    print("Multi-Experiment Optimization Results")
    print(f"{'='*60}\n")

    print(f"Solver Status: {doe_obj.results['Solver Status']}")
    print(f"Termination Condition: {doe_obj.results['Termination Condition']}")
    print(f"Number of Scenarios: {doe_obj.results['Number of Scenarios']}")
    print(
        f"Number of Experiments per Scenario: {doe_obj.results['Number of Experiments per Scenario']}"
    )

    print(f"\nTiming:")
    print(f"  Build Time: {doe_obj.results['Build Time']:.2f} seconds")
    print(
        f"  Initialization Time: {doe_obj.results['Initialization Time']:.2f} seconds"
    )
    print(f"  Solve Time: {doe_obj.results['Solve Time']:.2f} seconds")
    print(f"  Total Time: {doe_obj.results['Wall-clock Time']:.2f} seconds")

    # Print scenario-specific results
    for s_idx, scenario in enumerate(doe_obj.results['Scenarios']):
        print(f"\n{'-'*60}")
        print(f"Scenario {s_idx} Results:")
        print(f"{'-'*60}")

        print(f"\nAggregated FIM Statistics:")
        print(f"  log10 A-opt: {scenario['log10 A-opt']:.4f}")
        print(f"  log10 D-opt: {scenario['log10 D-opt']:.4f}")
        print(f"  log10 E-opt: {scenario['log10 E-opt']:.4f}")
        print(f"  log10 ME-opt: {scenario['log10 ME-opt']:.4f}")

        # Print each experiment design
        for exp_idx, exp in enumerate(scenario['Experiments']):
            print(f"\n  Experiment {exp_idx}:")
            print(f"    Design Variables:")
            for name, value in zip(
                doe_obj.results['Experiment Design Names'], exp['Experiment Design']
            ):
                print(f"      {name}: {value:.4f}")

    print(f"\n{'='*60}\n")

    return doe_obj


def run_multistart_doe(
    starting_hours,
    theta,
    measurement_error,
    objective_option="determinant",
    prior_FIM=None,
    tee=False,
):
    """
    Run multi-start optimization to escape local optima.

    Tries every combination of initial hour values, runs optimize_experiments
    from each starting point, and returns the best result found.

    Parameters
    ----------
    starting_hours : list of float
        Grid of hour values to use as starting points for each experiment.
        All 2-combinations (with replacement) are tried.
    theta : dict
        Parameter nominal values.
    measurement_error : float
        Measurement error magnitude.
    objective_option : str
        One of 'determinant', 'trace', 'pseudo_trace'.
    prior_FIM : np.ndarray, optional
        Prior FIM.
    tee : bool
        Show IPOPT output.

    Returns
    -------
    best_doe : DesignOfExperiments
        DoE object for the best start found.
    best_hours : tuple
        (h1_start, h2_start) of the winning starting point.
    summary : list of dict
        Objective value and starting hours for every trial.
    """
    from itertools import combinations_with_replacement

    # Map objective to the metric we compare (D-opt maximise, A-opt minimise,
    # pseudo_trace maximise).
    maximise = objective_option in ('determinant', 'pseudo_trace')
    metric_key = {
        'determinant': 'log10 D-opt',
        'trace': 'log10 A-opt',
        'pseudo_trace': 'log10 pseudo A-opt',
    }[objective_option]

    best_doe = None
    best_val = None
    best_hours = None
    summary = []
    n_trials = 0

    trials = list(combinations_with_replacement(starting_hours, 2))
    print(f"\n{'#'*70}")
    print(f"Multi-start DOE | objective={objective_option} | {len(trials)} starts")
    print(f"{'#'*70}")

    for h1_init, h2_init in trials:
        n_trials += 1
        print(f"\n--- Trial {n_trials}/{len(trials)}: init h1={h1_init:.2f}, h2={h2_init:.2f} ---")

        # Build experiment list initialised at (h1_init, h2_init)
        exp_list = []
        for h_init in (h1_init, h2_init):
            init_data = pd.Series({'hour': h_init, 'y': 10.0})
            exp_list.append(
                RooneyBieglerExperiment(
                    data=init_data, theta=theta, measure_error=measurement_error
                )
            )

        try:
            doe_obj = DesignOfExperiments(
                experiment_list=exp_list,
                objective_option=objective_option,
                prior_FIM=prior_FIM,
                tee=tee,
                _Cholesky_option=True,
                _only_compute_fim_lower=True,
            )
            doe_obj.optimize_experiments()

            # Extract objective value from first scenario
            scenario = doe_obj.results['Scenarios'][0]
            val = scenario.get(metric_key)

            if val is None:
                print(f"  Could not extract metric from scenario keys: {list(scenario.keys())}")
                summary.append({'h1_init': h1_init, 'h2_init': h2_init, 'val': None, 'status': 'metric_missing'})
                continue

            # Retrieve the solved hour values
            exp_designs = [
                exp['Experiment Design'][0]
                for exp in scenario['Experiments']
            ]

            print(f"  Solved: h1={exp_designs[0]:.4f}, h2={exp_designs[1]:.4f}, metric={val:.4f}")
            summary.append({'h1_init': h1_init, 'h2_init': h2_init, 'val': val,
                            'h1_opt': exp_designs[0], 'h2_opt': exp_designs[1],
                            'status': str(doe_obj.results['Termination Condition'])})

            is_better = (
                best_val is None
                or (maximise and val > best_val)
                or (not maximise and val < best_val)
            )
            if is_better:
                best_val = val
                best_doe = doe_obj
                best_hours = (h1_init, h2_init)

        except Exception as e:
            print(f"  Failed: {e}")
            summary.append({'h1_init': h1_init, 'h2_init': h2_init, 'val': None, 'status': f'error: {e}'})

    print(f"\n{'='*70}")
    print(f"Multi-start complete | best metric={best_val:.4f} | from start={best_hours}")
    if best_doe is not None:
        best_scen = best_doe.results['Scenarios'][0]
        best_designs = [exp['Experiment Design'][0] for exp in best_scen['Experiments']]
        print(f"Best design: h1={best_designs[0]:.4f}, h2={best_designs[1]:.4f}")
    print(f"{'='*70}\n")

    return best_doe, best_hours, summary


if __name__ == "__main__":
    # Data Setup
    data = pd.DataFrame(
        data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
        columns=['hour', 'y'],
    )
    theta = {'asymptote': 15, 'rate_constant': 0.5}
    measurement_error = 0.1

    # Create solver for initialization
    solver = pyo.SolverFactory("ipopt")
    # Use default linear solver

    # Test with 2 experiments
    print("\n" + "=" * 60)
    print("Multi-Experiment Optimization Test")
    print("=" * 60)

    # Objective options to test
    objective_options = ['determinant', 'trace', 'pseudo_trace']

    # Get script directory for saving results
    script_dir = Path(__file__).parent

    # Dictionary to store all results
    all_results = {}

    # Use no prior FIM for this simple example
    # Using prior from existing data can cause numerical issues
    p_FIM = np.zeros((2, 2))
    for i in range(len(data)):
        exp_data = data.loc[i, :]
        exp = RooneyBieglerExperiment(
            data=exp_data, theta=theta, measure_error=measurement_error
        )
        doe_data = DesignOfExperiments(experiment_list=[exp], step=0.01)
        p_FIM += doe_data.compute_FIM()

    # Starting hours for multi-start: 5 evenly spaced points across [0.5, 9.5]
    # so the optimizer can explore both the low-hour and high-hour basins.
    starting_hours = [0.5, 2.5, 5.0, 7.5, 9.5]

    comparison = {}  # objective -> {single: ..., multistart: ...}

    for obj_option in objective_options:
        print(f"\n\n{'#'*70}")
        print(f"# Objective: {obj_option}")
        print(f"{'#'*70}")

        # ---- Single-start (original behaviour, init at hour=1 and hour=2) ----
        print(f"\n--- Single-start (init: hour=1, hour=2) ---")
        single_exp_list = []
        for i in range(2):
            exp_data = data.loc[i, :]
            single_exp_list.append(
                RooneyBieglerExperiment(
                    data=exp_data, theta=theta, measure_error=measurement_error
                )
            )
        doe_single = run_rooney_biegler_multi_experiment_doe(
            experiment_list=single_exp_list,
            objective_option=obj_option,
            prior_FIM=p_FIM,
            tee=False,
        )
        single_scenario = doe_single.results['Scenarios'][0]
        single_designs = [
            exp['Experiment Design'][0] for exp in single_scenario['Experiments']
        ]
        single_d_opt = single_scenario['log10 D-opt']
        single_a_opt = single_scenario['log10 A-opt']

        # ---- Multi-start ----
        best_doe, best_start, ms_summary = run_multistart_doe(
            starting_hours=starting_hours,
            theta=theta,
            measurement_error=measurement_error,
            objective_option=obj_option,
            prior_FIM=p_FIM,
            tee=False,
        )
        ms_scenario = best_doe.results['Scenarios'][0]
        ms_designs = [
            exp['Experiment Design'][0] for exp in ms_scenario['Experiments']
        ]
        ms_d_opt = ms_scenario['log10 D-opt']
        ms_a_opt = ms_scenario['log10 A-opt']

        comparison[obj_option] = {
            'single_start': {
                'init_hours': [1.0, 2.0],
                'opt_hours': single_designs,
                'log10_D_opt': single_d_opt,
                'log10_A_opt': single_a_opt,
            },
            'multi_start': {
                'best_start': list(best_start),
                'opt_hours': ms_designs,
                'log10_D_opt': ms_d_opt,
                'log10_A_opt': ms_a_opt,
                'all_trials': ms_summary,
            },
        }

        print(f"\n  Single-start  → h=({single_designs[0]:.4f}, {single_designs[1]:.4f})"
              f"  D-opt={single_d_opt:.4f}  A-opt={single_a_opt:.4f}")
        print(f"  Multi-start   → h=({ms_designs[0]:.4f}, {ms_designs[1]:.4f})"
              f"  D-opt={ms_d_opt:.4f}  A-opt={ms_a_opt:.4f}")

        # Save best (multi-start) result as the canonical JSON for this objective
        results_summary = {
            'objective_option': obj_option,
            'method': 'multi_start',
            'starting_hours': starting_hours,
            'best_start': list(best_start),
            'solver_status': str(best_doe.results['Solver Status']),
            'termination_condition': str(best_doe.results['Termination Condition']),
            'n_scenarios': best_doe.results['Number of Scenarios'],
            'n_experiments': best_doe.results['Number of Experiments per Scenario'],
            'timing': {
                'build_time': best_doe.results['Build Time'],
                'initialization_time': best_doe.results['Initialization Time'],
                'solve_time': best_doe.results['Solve Time'],
                'total_time': best_doe.results['Wall-clock Time'],
            },
            'scenarios': [],
        }
        for s_idx, scenario in enumerate(best_doe.results['Scenarios']):
            scenario_data = {
                'scenario_idx': s_idx,
                'log10_A_opt': scenario['log10 A-opt'],
                'log10_D_opt': scenario['log10 D-opt'],
                'log10_E_opt': scenario['log10 E-opt'],
                'FIM_condition_number': scenario['FIM Condition Number'],
                'experiments': [],
            }
            for exp_idx, exp in enumerate(scenario['Experiments']):
                exp_entry = {'experiment_idx': exp_idx, 'design_variables': {}}
                for name, value in zip(
                    best_doe.results['Experiment Design Names'], exp['Experiment Design']
                ):
                    exp_entry['design_variables'][name] = value
                scenario_data['experiments'].append(exp_entry)
            results_summary['scenarios'].append(scenario_data)

        all_results[obj_option] = results_summary
        output_file = script_dir / f'rooney_biegler_multiexp_{obj_option}.json'
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"  Results saved to: {output_file}")

    # Save combined results
    combined_file = script_dir / 'rooney_biegler_multiexp_all_objectives.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nCombined results saved to: {combined_file}")

    # Print final comparison table
    print("\n" + "=" * 70)
    print("Single-start vs Multi-start Comparison")
    print("=" * 70)
    print(f"{'Objective':<15} {'Method':<14} {'h1':>8} {'h2':>8} {'D-opt':>10} {'A-opt':>10}")
    print("-" * 70)
    for obj, vals in comparison.items():
        ss = vals['single_start']
        ms = vals['multi_start']
        print(f"{obj:<15} {'single-start':<14} {ss['opt_hours'][0]:>8.4f} {ss['opt_hours'][1]:>8.4f}"
              f" {ss['log10_D_opt']:>10.4f} {ss['log10_A_opt']:>10.4f}")
        print(f"{'':15} {'multi-start':<14} {ms['opt_hours'][0]:>8.4f} {ms['opt_hours'][1]:>8.4f}"
              f" {ms['log10_D_opt']:>10.4f} {ms['log10_A_opt']:>10.4f}")
        print("-" * 70)

    print("\nAll tests completed successfully!")
