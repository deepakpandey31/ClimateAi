"""
optimizer.py — Budget-constrained intervention optimizer using PuLP.

Solves a linear program to maximize total city-wide LST reduction
across detected hotspots, subject to a user-specified resource budget.

Objective is weighted by population exposure (HVI) — so the optimizer
doesn't dump the entire budget on the single hottest cell if a cooler
but more densely populated cell benefits more people.

The allocation changes meaningfully when the budget changes — no
hardcoded "always pick the hottest cell" heuristic.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def optimize_budget(
    simulation_df: pd.DataFrame,
    budget: float,
    budget_type: str = 'saplings',
    pop_weights: Optional[Dict[int, float]] = None,
    min_intensity: float = 1.0,
    max_intensity: float = 10.0,
) -> Dict[str, Any]:
    """
    Allocate a fixed resource budget across hotspots and interventions
    to maximize population-weighted citywide LST reduction.

    Args:
        simulation_df: Output from intervention_simulator.batch_simulate()
                       Must have columns: cell_id, intervention_type,
                       delta_lst_c, resource_amount, resource_unit,
                       locality_name, pop_density, lst_anomaly_c
        budget: Total available resource quantity
        budget_type: One of 'saplings', 'cool_roof_m2', 'rupees'
        pop_weights: Optional dict {cell_id: population_weight} for HVI-based weighting
                     If None, weights are proportional to pop_density in simulation_df
        min_intensity: Minimum intensity (1–10) for any selected intervention
        max_intensity: Maximum intensity (1–10) for any selected intervention

    Returns:
        dict with:
            allocations: list of {cell_id, locality, intervention, intensity, delta_lst, cost}
            total_cost: total resource used
            total_delta_lst_c: total predicted cooling (city-wide, population-weighted)
            unweighted_delta_lst_c: simple sum of predicted cooling
            budget_utilization: fraction of budget used
            solver_status: 'optimal', 'feasible', 'infeasible', 'fallback'
    """
    try:
        import pulp
    except ImportError:
        logger.error("PuLP not installed. Using greedy fallback optimizer.")
        return _greedy_optimizer(simulation_df, budget, budget_type, pop_weights)

    # ── Map budget type to resource unit ──────────────────────────────────────
    BUDGET_TYPE_MAPPING = {
        'saplings': 'saplings',
        'cool_roof_m2': 'm² material',
        'green_roof_m2': 'm² installation',
        'water_ha': 'hectares',
        'pavement_m2': 'm² pavement',
        'rupees': None,  # special: convert all costs to rupees
    }

    COST_PER_UNIT = {  # Cost in ₹ per resource unit (for ₹ budget mode)
        'saplings': 200,          # ₹200 per sapling (includes labor)
        'm² material': 800,       # ₹800/m² cool-roof material + labor
        'm² installation': 2500,  # ₹2500/m² green roof
        'hectares': 500000,       # ₹5 lakh per hectare water body
        'm² pavement': 1200,      # ₹1200/m² reflective pavement
    }

    if simulation_df.empty:
        logger.warning("Empty simulation_df — no interventions to optimize.")
        return _empty_result()

    # Filter to matching resource unit for this budget type
    if budget_type == 'rupees':
        sim = simulation_df.copy()
        sim['cost'] = sim.apply(
            lambda row: row['resource_amount'] * COST_PER_UNIT.get(row['resource_unit'], 1000),
            axis=1
        )
    else:
        target_unit = BUDGET_TYPE_MAPPING.get(budget_type, 'saplings')
        sim = simulation_df[simulation_df['resource_unit'] == target_unit].copy()
        if sim.empty:
            # Fallback: use all interventions
            sim = simulation_df.copy()
            logger.warning(f"No interventions with resource_unit '{target_unit}'. Using all.")
        sim['cost'] = sim['resource_amount']

    # ── Compute population weights ─────────────────────────────────────────────
    if pop_weights is None:
        # Normalize pop_density as weight
        max_pop = sim['pop_density'].max()
        if max_pop > 0:
            sim['weight'] = 0.5 + 0.5 * (sim['pop_density'] / max_pop)
        else:
            sim['weight'] = 1.0
    else:
        sim['weight'] = sim['cell_id'].map(pop_weights).fillna(1.0)

    # Also weight by LST anomaly severity
    max_anomaly = sim['lst_anomaly_c'].abs().max()
    if max_anomaly > 0:
        sim['anomaly_weight'] = 0.5 + 0.5 * (sim['lst_anomaly_c'].abs() / max_anomaly)
    else:
        sim['anomaly_weight'] = 1.0

    # Combined weight: equal mix of population and anomaly severity
    sim['combined_weight'] = (sim['weight'] + sim['anomaly_weight']) / 2.0

    # Weighted cooling benefit (what we maximize)
    # delta_lst_c is negative (cooling), so we minimize negative = maximize cooling
    sim['weighted_benefit'] = -sim['delta_lst_c'] * sim['combined_weight']

    # ── Build LP ───────────────────────────────────────────────────────────────
    # Decision variables: x_i ∈ [0, 1] = whether to select intervention i
    # (Relaxed LP — fractional allocation allowed for budget efficiency)
    # For simplicity: binary selection at max intensity per cell (one intervention per cell)

    prob = pulp.LpProblem("UrbanHeatMitigation", pulp.LpMaximize)

    n = len(sim)
    x = [pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Continuous') for i in range(n)]

    # Objective: maximize weighted cooling benefit
    prob += pulp.lpSum(sim['weighted_benefit'].iloc[i] * x[i] for i in range(n))

    # Budget constraint
    prob += pulp.lpSum(sim['cost'].iloc[i] * x[i] for i in range(n)) <= budget, "budget"

    # One intervention type per cell (can't do everything to one place)
    cell_ids = sim['cell_id'].unique()
    for cell_id in cell_ids:
        cell_rows = sim[sim['cell_id'] == cell_id]
        cell_indices = [sim.index.get_loc(idx) for idx in cell_rows.index]
        prob += pulp.lpSum(x[i] for i in cell_indices) <= 1.0, f"one_per_cell_{cell_id}"

    # Solve
    try:
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
        prob.solve(solver)
        solver_status = pulp.LpStatus[prob.status]
    except Exception as e:
        logger.warning(f"PuLP solver error: {e}. Using greedy fallback.")
        return _greedy_optimizer(simulation_df, budget, budget_type, pop_weights)

    if prob.status not in [1, -1]:  # 1=Optimal, -1=Infeasible
        logger.warning(f"LP solver status: {solver_status}. Using greedy fallback.")
        return _greedy_optimizer(simulation_df, budget, budget_type, pop_weights)

    # ── Extract solution ───────────────────────────────────────────────────────
    allocations = []
    total_cost = 0.0
    total_cooling = 0.0
    unweighted_cooling = 0.0

    for i in range(n):
        x_val = pulp.value(x[i]) or 0.0
        if x_val > 0.01:  # Significant allocation
            row = sim.iloc[i]
            cost = float(row['cost']) * x_val
            cooling = float(row['delta_lst_c']) * x_val

            allocations.append({
                'cell_id': int(row['cell_id']),
                'locality_name': row.get('locality_name', f"Cell {row['cell_id']}"),
                'intervention_type': row['intervention_type'],
                'intervention_label': row.get('intervention_label', row['intervention_type']),
                'allocation_fraction': round(x_val, 3),
                'effective_intensity': round(float(row['intensity']) * x_val, 1),
                'delta_lst_c': round(cooling, 2),
                'delta_lst_lower_c': round(float(row.get('delta_lst_lower_c', cooling * 1.2)), 2),
                'delta_lst_upper_c': round(float(row.get('delta_lst_upper_c', cooling * 0.8)), 2),
                'cost': round(cost, 0),
                'resource_unit': row['resource_unit'],
                'weighted_benefit': round(float(row['weighted_benefit']) * x_val, 3),
                'pop_weight': round(float(row['weight']), 3),
                'lst_anomaly_c': float(row['lst_anomaly_c']),
            })

            total_cost += cost
            total_cooling += cooling * float(row['combined_weight'])
            unweighted_cooling += cooling

    logger.info(
        f"Optimizer: {len(allocations)} interventions selected, "
        f"total cost={total_cost:.0f}, "
        f"predicted total cooling={unweighted_cooling:.2f}°C"
    )

    return {
        'allocations': sorted(allocations, key=lambda x: x['delta_lst_c']),
        'total_cost': round(total_cost, 0),
        'budget': budget,
        'budget_type': budget_type,
        'total_delta_lst_c': round(unweighted_cooling, 2),
        'weighted_delta_lst_c': round(total_cooling, 2),
        'budget_utilization': round(total_cost / budget, 3) if budget > 0 else 0,
        'solver_status': solver_status,
        'n_hotspots_addressed': len(set(a['cell_id'] for a in allocations)),
    }


def _greedy_optimizer(
    simulation_df: pd.DataFrame,
    budget: float,
    budget_type: str,
    pop_weights: Optional[Dict],
) -> Dict[str, Any]:
    """
    Greedy fallback optimizer (used if PuLP fails).
    Selects highest-benefit-per-cost interventions greedily.
    """
    logger.info("Using greedy fallback optimizer")
    if simulation_df.empty:
        return _empty_result()

    sim = simulation_df.copy()
    sim['cost'] = sim['resource_amount']
    sim['benefit_per_cost'] = (-sim['delta_lst_c']) / (sim['cost'] + 1)

    sim = sim.sort_values('benefit_per_cost', ascending=False)
    remaining_budget = budget
    allocations = []
    used_cells = set()
    total_cooling = 0.0

    for _, row in sim.iterrows():
        cell_id = int(row['cell_id'])
        if cell_id in used_cells:
            continue
        if row['cost'] > remaining_budget:
            continue

        allocations.append({
            'cell_id': cell_id,
            'locality_name': row.get('locality_name', f"Cell {cell_id}"),
            'intervention_type': row['intervention_type'],
            'intervention_label': row.get('intervention_label', row['intervention_type']),
            'allocation_fraction': 1.0,
            'effective_intensity': float(row['intensity']),
            'delta_lst_c': round(float(row['delta_lst_c']), 2),
            'delta_lst_lower_c': round(float(row.get('delta_lst_lower_c', row['delta_lst_c'] * 1.2)), 2),
            'delta_lst_upper_c': round(float(row.get('delta_lst_upper_c', row['delta_lst_c'] * 0.8)), 2),
            'cost': round(float(row['cost']), 0),
            'resource_unit': row['resource_unit'],
            'lst_anomaly_c': float(row.get('lst_anomaly_c', 0)),
        })

        remaining_budget -= row['cost']
        used_cells.add(cell_id)
        total_cooling += float(row['delta_lst_c'])

    return {
        'allocations': allocations,
        'total_cost': round(budget - remaining_budget, 0),
        'budget': budget,
        'budget_type': budget_type,
        'total_delta_lst_c': round(total_cooling, 2),
        'weighted_delta_lst_c': round(total_cooling, 2),
        'budget_utilization': round((budget - remaining_budget) / budget, 3) if budget > 0 else 0,
        'solver_status': 'greedy_fallback',
        'n_hotspots_addressed': len(allocations),
    }


def _empty_result() -> Dict:
    return {
        'allocations': [],
        'total_cost': 0,
        'budget': 0,
        'budget_type': 'unknown',
        'total_delta_lst_c': 0.0,
        'weighted_delta_lst_c': 0.0,
        'budget_utilization': 0.0,
        'solver_status': 'no_data',
        'n_hotspots_addressed': 0,
    }
