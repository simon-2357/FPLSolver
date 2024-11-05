import pandas as pd
import pulp as pl
import numpy as np
from helpers import add_noise
from typing import List, Dict
from dataclasses import dataclass
import os



@dataclass
class OptimizationConfig:
    data_input: str
    ft_settings: dict
    sub_weights: dict
    exact_money_penalty: dict
    chip_weeks: dict
    transfer_position: dict
    initial_squad: List[int]
    sensitivity_settings: dict
    horizon_settings: dict
    objective_settings: dict
    change_ev: List[int]
    reduce_availability: List[int]
    undecay_ev: bool
    force_next_gw: dict
    force_horizon: dict
    force_particular_gw: dict
    priority: List[int]
    priority_plus: List[int]
    soft_winner: List[int]
    buyback_likelihood: List[int]
    price_drop_rate: List[int]
    force_roll: List[int]
    tied_up_value: float
    output_options: dict


def load_data(config: OptimizationConfig) -> pd.DataFrame:
    df = pd.read_csv(f'data/{config.data_input}.csv')
    df.set_index('ID', inplace=True)
    return df       


def create_player_lists(data: pd.DataFrame) -> Dict[str, List[int]]:
    pos_group = data.groupby(data['Pos'])
    team_group = data.groupby(data['Team'])
    
    player_lists = {
        'players': data.index.tolist(),
        'goalkeepers': pos_group.get_group('G').index.tolist(),
        'defenders': pos_group.get_group('D').index.tolist(),
        'midfielders': pos_group.get_group('M').index.tolist(),
        'forwards': pos_group.get_group('F').index.tolist(),
    }
    
    team_names = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton', 'Fulham', 'Brentford', 'Chelsea',
                  'Crystal Palace', 'Everton', 'Ipswich', 'Leicester', 'Liverpool', 'Man City',
                  'Man Utd', 'Newcastle', 'Southampton', 'Spurs',
                  "Nott'm Forest", 'West Ham', 'Wolves']
    
    for team in team_names:
        key = team.lower().replace(' ', '_').replace("'", '')
        player_lists[key] = team_group.get_group(team).index.tolist()
    
    return player_lists


def initialize_model(config: OptimizationConfig, data: pd.DataFrame, player_lists: Dict[str, List[int]]):
    model = pl.LpProblem('model', pl.LpMaximize)
    next_gw = int(data.keys()[6].split('_')[0]) + config.horizon_settings["future_week"]
    gameweeks = list(range(next_gw, next_gw + config.horizon_settings["horizon"]))
    all_gw = [next_gw - 1] + gameweeks
    gwminus = list(range(next_gw, next_gw + config.horizon_settings["horizon"] - 1))
    
    if config.reduce_availability:
        for item in config.reduce_availability:
            for w in item[2]:
                data.loc[item[0], [f'{w}_xMins']] *= item[1]
                data.loc[item[0], [f'{w}_Pts']] *= item[1]

    data['tv_loss_sell_penalty'] = (data['BV'] - data['SV']) * config.tied_up_value * (39 - next_gw - config.horizon_settings["horizon"])
    data['hold_penalty'] = 0

    if config.price_drop_rate:
        for item in config.price_drop_rate:
            data.loc[item[0], ['hold_penalty']] += config.tied_up_value * (39 - next_gw - config.horizon_settings["horizon"]) * item[1]

    if config.undecay_ev:
        for w in gameweeks:
            data[f'{w}_Pts'] = data[f'{w}_Pts'] * (1.017) ** (w-next_gw)

    if config.buyback_likelihood:
        for item in config.buyback_likelihood:
            data.loc[item[0], ['tv_loss_sell_penalty']] *= item[1]

    if config.horizon_settings["compress"] > 0:
        points_columns = [f'{gw}_Pts' for gw in range (next_gw + config.horizon_settings["csv_weeks"] - config.horizon_settings["compress"] - config.horizon_settings["future_week"], next_gw + config.horizon_settings["csv_weeks"] - config.horizon_settings["future_week"])]
        xmin_columns = [f'{gw}_xMins' for gw in range (next_gw + config.horizon_settings["csv_weeks"] - config.horizon_settings["compress"] - config.horizon_settings["future_week"], next_gw + config.horizon_settings["csv_weeks"] - config.horizon_settings["future_week"])]
        decay_weights = [config.objective_settings["decay"] ** i for i in range(config.horizon_settings["compress"])]
        total_weight = sum(decay_weights)
        decay_weights = [w / total_weight for w in decay_weights]
        data['xMins Avg'] = data[xmin_columns].mul(decay_weights).sum(axis=1)
        data['Points Avg'] = data[points_columns].mul(decay_weights).sum(axis=1)
        points_target_column = points_columns[0]
        xmin_target_column = xmin_columns[0]
        data[points_target_column] = data['Points Avg'] * (1 - config.horizon_settings["compress_sum"] + config.horizon_settings["compress_sum"] * total_weight)
        data[xmin_target_column] = data['xMins Avg']
        data.drop(columns=['Points Avg', 'xMins Avg'], inplace=True)

    if config.change_ev:
        for item in config.change_ev:
            for w in gameweeks:
                data.loc[item[0], [f'{w}_Pts']] *= 1 + item[1]/100

    if config.chip_weeks['fh'] in gameweeks:
        gameweeks = gwminus
        horizon = horizon - 1
        if config.chip_weeks['fh'] < config.chip_weeks['wc']:
            config.chip_weeks['wc'] = config.chip_weeks['wc'] - 1
        if config.chip_weeks['fh'] < config.chip_weeks['bb']:
            config.chip_weeks['bb'] = config.chip_weeks['bb'] - 1
        for w in gwminus:
            if w >= config.chip_weeks['fh']:
                data[f'{w}_Pts'] = data[f'{w + 1}_Pts']

    rng = np.random.default_rng()
    data.loc[data['Pos'] == 'G', ['pos_noise']] = -0.0176
    data.loc[data['Pos'] == 'D', ['pos_noise']] = 0
    data.loc[data['Pos'] == 'M', ['pos_noise']] = -0.0553
    data.loc[data['Pos'] == 'F', ['pos_noise']] = -0.0414
    
    for w in gameweeks:
        noise = (0.7293 + data[f'{w}_Pts'] * 0.0044 - data[f'{w}_xMins'] * 0.0083 + (w - next_gw) * 0.0092 + data['pos_noise']) * rng.standard_normal(size=len(player_lists['players'])) * config.sensitivity_settings["magnitude"]
        data[f'{w}_Pts'] = data[f'{w}_Pts'] * (1 + noise)
    
    return model, next_gw, gameweeks, all_gw, data

def create_decision_variables(players: List[int], gameweeks: List[int], all_gw: List[int]) -> Dict:
    vars = {
        'lineup': pl.LpVariable.dicts('lineup', (players, gameweeks), 0, 1, cat='Integer'),
        'squad': pl.LpVariable.dicts('squad', (players, all_gw), 0, 1, cat='Integer'),
        'bench1': pl.LpVariable.dicts('bench1', (players, gameweeks), 0, 1, cat='Integer'),
        'bench2': pl.LpVariable.dicts('bench2', (players, gameweeks), 0, 1, cat='Integer'),
        'bench3': pl.LpVariable.dicts('bench3', (players, gameweeks), 0, 1, cat='Integer'),
        'captain': pl.LpVariable.dicts('captain', (players, gameweeks), 0, 1, cat='Integer'),
        'vicecap': pl.LpVariable.dicts('vicecap', (players, gameweeks), 0, 1, cat='Integer'),
        'transfer_in': pl.LpVariable.dicts('transfer_in', (players, gameweeks), 0, 1, cat='Integer'),
        'transfer_out': pl.LpVariable.dicts('transfer_out', (players, gameweeks), 0, 1, cat='Integer'),
        'in_the_bank': pl.LpVariable.dicts('itb', all_gw, 0),
        'free_transfers': pl.LpVariable.dicts('ft', all_gw, 0, 5, cat='Integer'),
        'hits': pl.LpVariable.dicts('hits', gameweeks, 0, cat='Integer'),
        'carry': pl.LpVariable.dicts('carry', all_gw, 0, 4, cat='Integer'),
        'use_bb': pl.LpVariable.dicts('use_bb', gameweeks, 0, 1, cat='Integer'),
        'use_wc': pl.LpVariable.dicts('use_wc', gameweeks, 0, 1, cat='Integer'),
        'use_tc': pl.LpVariable.dicts('use_tc', gameweeks, 0, 1, cat='Integer'),
        'carry_bin': {(w, i): pl.LpVariable(f'carry_bin_{w}_{i}', cat='Binary') for w in gameweeks for i in range(5)},  
        'transfer_indicator': {w: pl.LpVariable(f"transfer_indicator_{w}", cat='Binary') for w in gameweeks},
    }
    return vars

def add_constraints(model: pl.LpProblem, config: OptimizationConfig, data: pd.DataFrame, 
                    player_lists: Dict[str, List[int]], vars: Dict, bank_potential_penalty: Dict):
    next_gw = int(data.columns[6].split('_')[0]) + config.horizon_settings["future_week"]
    gameweeks = list(range(next_gw, next_gw + config.horizon_settings["horizon"]))
    all_gw = [next_gw - 1] + gameweeks

    # Initial conditions
    for p in player_lists['players']:
        if p in config.initial_squad:
            vars['squad'][p][next_gw-1] = 1
        else:
            vars['squad'][p][next_gw-1] = 0

    vars['in_the_bank'][next_gw-1] = config.transfer_position["money_itb"]
    vars['carry'][next_gw-1] = config.transfer_position["ft_available"] - 1

    # Chip usage
    all_gw_plus = all_gw + [next_gw + config.horizon_settings["horizon"]]
    for w in all_gw_plus:
        if w == config.chip_weeks['bb']:
            vars['use_bb'][w] = 1
        else:
            vars['use_bb'][w] = 0
        if w == config.chip_weeks['wc']:
            vars['use_wc'][w] = 1
        else:
            vars['use_wc'][w] = 0
        if w == config.chip_weeks['tc']:
            vars['use_tc'][w] = 1
        else:
            vars['use_tc'][w] = 0

    if config.chip_weeks['wc'] in gameweeks:
        model += vars['carry'][config.chip_weeks['wc']-1] == vars['carry'][config.chip_weeks['wc']]

    # Player constraints
    for p in config.force_next_gw["ban"]:
        model += vars['squad'][p][next_gw] == 0
    for p in config.force_next_gw["lock"]:
        model += vars['squad'][p][next_gw] == 1
    for w in gameweeks:
        for p in config.force_horizon["ban"]:
            model += vars['squad'][p][w] == 0
    for w in gameweeks:
        for p in config.force_horizon["lock"]:
            model += vars['squad'][p][w] == 1

    # Transfer constraints
    for w in config.force_roll:
        model += pl.lpSum(0.5 * (vars['transfer_in'][p][w] + vars['transfer_out'][p][w]) for p in player_lists['players']) == vars['hits'][w]
        
    if config.priority:
        for item in config.priority:
            for winner in item[0]:
                for loser in item[1]:
                    model += vars['squad'][winner][next_gw] >= vars['squad'][loser][next_gw]
    if config.priority_plus:
        for item in config.priority_plus:
            for winner in item[0]:
                for loser in item[1]:
                    model += vars['squad'][winner][next_gw+1] >= vars['squad'][loser][next_gw+1]
    if config.soft_winner:
        for item in config.soft_winner:
                if item[0][2]:
                    model += vars['squad'][item[0][0]][next_gw] + vars['squad'][item[0][1]][next_gw] + + vars['squad'][item[0][2]][next_gw] >= vars['squad'][item[1][0]][next_gw]
                else:
                    model += vars['squad'][item[0][0]][next_gw] + vars['squad'][item[0][1]][next_gw] >= vars['squad'][item[1][0]][next_gw]

    for item in config.force_particular_gw["lock"]:
        model += vars['squad'][item[0]][item[1]] == 1 
    for item in config.force_particular_gw["ban"]:
        model += vars['squad'][item[0]][item[1]] == 0

    # Squad mechanics
    for w in gameweeks:
        model += vars['in_the_bank'][w] - vars['in_the_bank'][w-1] == pl.lpSum(data['SV'][p] * vars['transfer_out'][p][w] for p in player_lists['players']) - pl.lpSum(data['BV'][p] * vars['transfer_in'][p][w] for p in player_lists['players'])
        model += vars['in_the_bank'][w] >= 0
        model += vars['free_transfers'][w] == vars['carry'][w-1] + 1 - vars['use_wc'][w]
        model += vars['free_transfers'][w] - (1-vars['use_wc'][w]) * (pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players'])) + vars['hits'][w] >= vars['carry'][w]
        model += pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players']) <= 15
        model += vars['carry'][w] <= vars['carry'][w-1] + 1 - vars['use_wc'][w]
        model += vars['hits'][w] >= (1-vars['use_wc'][w]) * (pl.lpSum(0.5 * (vars['transfer_in'][p][w] + vars['transfer_out'][p][w]) for p in player_lists['players'])) - vars['free_transfers'][w]
        model += pl.lpSum(vars['carry_bin'][w, i] for i in range(5)) == 1
        model += vars['carry'][w] == pl.lpSum(i * vars['carry_bin'][w, i] for i in range(5))
        model += vars['transfer_indicator'][w] <= pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players'])
        model += pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players']) <= 100 * vars['transfer_indicator'][w]
        model += pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players']) >= 0
        model += pl.lpSum(vars['transfer_out'][p][w] for p in player_lists['players']) >= 0
        model += pl.lpSum(vars['transfer_in'][p][w] for p in player_lists['players']) == pl.lpSum(vars['transfer_out'][p][w] for p in player_lists['players'])

        for p in player_lists['players']:
            model += vars['transfer_in'][p][w] >= 0
            model += vars['transfer_out'][p][w] >= 0
            model += vars['squad'][p][w] - vars['squad'][p][w - 1] == vars['transfer_in'][p][w] - vars['transfer_out'][p][w]
            model += vars['transfer_in'][p][w] + vars['transfer_out'][p][w] <= 1

    # Squad formation constraints
    for w in gameweeks:
        model += pl.lpSum(vars['squad'][p][w] for p in player_lists['players']) == 15
        model += pl.lpSum(vars['squad'][g][w] for g in player_lists['goalkeepers']) == 2
        model += pl.lpSum(vars['squad'][d][w] for d in player_lists['defenders']) == 5
        model += pl.lpSum(vars['squad'][m][w] for m in player_lists['midfielders']) == 5
        model += pl.lpSum(vars['squad'][f][w] for f in player_lists['forwards']) == 3
        model += pl.lpSum(vars['lineup'][p][w] for p in player_lists['players']) == 11 + 4 * vars['use_bb'][w]
        model += pl.lpSum(vars['bench1'][p][w] for p in player_lists['players']) == 1 - vars['use_bb'][w]
        model += pl.lpSum(vars['bench2'][p][w] for p in player_lists['players']) == 1 - vars['use_bb'][w]
        model += pl.lpSum(vars['bench3'][p][w] for p in player_lists['players']) == 1 - vars['use_bb'][w]
        model += pl.lpSum(vars['bench1'][g][w] for g in player_lists['goalkeepers']) == 0
        model += pl.lpSum(vars['bench2'][g][w] for g in player_lists['goalkeepers']) == 0
        model += pl.lpSum(vars['bench3'][g][w] for g in player_lists['goalkeepers']) == 0
        model += pl.lpSum(vars['lineup'][g][w] for g in player_lists['goalkeepers']) == 1 + vars['use_bb'][w]
        model += pl.lpSum(vars['lineup'][d][w] for d in player_lists['defenders']) >= 3
        model += pl.lpSum(vars['lineup'][d][w] for d in player_lists['defenders']) <= 5
        model += pl.lpSum(vars['lineup'][m][w] for m in player_lists['midfielders']) >= 2
        model += pl.lpSum(vars['lineup'][m][w] for m in player_lists['midfielders']) <= 5
        model += pl.lpSum(vars['lineup'][f][w] for f in player_lists['forwards']) >= 1
        model += pl.lpSum(vars['lineup'][f][w] for f in player_lists['forwards']) <= 3
        model += pl.lpSum(vars['captain'][p][w] for p in player_lists['players']) == 1
        model += pl.lpSum(vars['vicecap'][p][w] for p in player_lists['players']) == 1

        # Team constraints
        for team in ['arsenal', 'aston_villa', 'brentford', 'brighton', 'nottm_forest', 'chelsea', 'crystal_palace', 'everton', 'leicester', 'ipswich', 'liverpool', 'man_city', 'man_utd', 'newcastle', 'bournemouth', 'southampton', 'spurs', 'fulham', 'west_ham', 'wolves']:
            model += pl.lpSum(vars['squad'][x][w] for x in player_lists[team]) <= 3

    # Lineup & Captain constraints
    for w in gameweeks:
        for p in player_lists['players']:
            model += vars['lineup'][p][w] <= vars['squad'][p][w]
            model += vars['captain'][p][w] <= vars['lineup'][p][w]
            model += vars['vicecap'][p][w] <= vars['lineup'][p][w]
            model += vars['vicecap'][p][w] + vars['captain'][p][w] <= 1
            model += vars['bench1'][p][w] <= vars['squad'][p][w]
            model += vars['bench2'][p][w] <= vars['squad'][p][w]
            model += vars['bench3'][p][w] <= vars['squad'][p][w]
            model += vars['bench1'][p][w] + vars['bench2'][p][w] + vars['bench3'][p][w] <= 1

def calculate_bank_potential_penalty(vars, gameweeks, config, model):
    exact_money_penalty = add_noise(variable_value=config.exact_money_penalty["0.0"], standard_deviation=config.exact_money_penalty["0.0_sd"], noise_on=config.sensitivity_settings["magnitude"])
    zero_point_one_penalty = add_noise(variable_value=config.exact_money_penalty["0.1"], standard_deviation=config.exact_money_penalty["0.1_sd"], noise_on=config.sensitivity_settings["magnitude"])
    zero_point_two_penalty = add_noise(variable_value=config.exact_money_penalty["0.2"], standard_deviation=config.exact_money_penalty["0.2_sd"], noise_on=config.sensitivity_settings["magnitude"])
    carry_reduction_factor = add_noise(variable_value=config.exact_money_penalty["carry_reduction"], standard_deviation=config.exact_money_penalty["carry_reduction_sd"], noise_on=config.sensitivity_settings["magnitude"])

    M = 1000  # A large number

    bank_potential_penalty = {}
    for w in gameweeks:
        # Create binary variables for each bank amount case
        is_zero = pl.LpVariable(f"is_zero_{w}", cat='Binary')
        is_point_one = pl.LpVariable(f"is_point_one_{w}", cat='Binary')
        is_point_two = pl.LpVariable(f"is_point_two_{w}", cat='Binary')

        # Constraints to set the binary variables
        model += vars['in_the_bank'][w] >= 0
        model += is_zero + is_point_one + is_point_two <= 1

        # Constraints for is_zero
        model += vars['in_the_bank'][w] <= (0 + M * (1 - is_zero))
        model += vars['in_the_bank'][w] >= (0 - M * (1 - is_zero))

        # Constraints for is_point_one
        model += vars['in_the_bank'][w] <= (0.1 + M * (1 - is_point_one))
        model += vars['in_the_bank'][w] >= (0.1 - M * (1 - is_point_one))

        # Constraints for is_point_two
        model += vars['in_the_bank'][w] <= (0.2 + M * (1 - is_point_two))
        model += vars['in_the_bank'][w] >= (0.2 - M * (1 - is_point_two))

        # Calculate the penalty
        bank_potential_penalty[w] = pl.LpVariable(f"bank_potential_penalty_{w}", lowBound=0)

        # Use separate variables for each term to avoid multiplication
        penalty_zero = pl.LpVariable(f"penalty_zero_{w}", lowBound=0)
        penalty_point_one = pl.LpVariable(f"penalty_point_one_{w}", lowBound=0)
        penalty_point_two = pl.LpVariable(f"penalty_point_two_{w}", lowBound=0)

        # Set up constraints for each penalty term
        model += penalty_zero <= M * is_zero
        model += penalty_zero <= exact_money_penalty * (w - gameweeks[0])
        model += penalty_zero >= exact_money_penalty * (w - gameweeks[0]) - M * (1 - is_zero)

        model += penalty_point_one <= M * is_point_one
        model += penalty_point_one <= zero_point_one_penalty * (w - gameweeks[0])
        model += penalty_point_one >= zero_point_one_penalty * (w - gameweeks[0]) - M * (1 - is_point_one)

        model += penalty_point_two <= M * is_point_two
        model += penalty_point_two <= zero_point_two_penalty * (w - gameweeks[0])
        model += penalty_point_two >= zero_point_two_penalty * (w - gameweeks[0]) - M * (1 - is_point_two)

        # Sum up the penalties
        model += bank_potential_penalty[w] == penalty_zero + penalty_point_one + penalty_point_two

        # Apply the carry decay
        for i in range(5):
            carry_penalty = pl.LpVariable(f"carry_penalty_{w}_{i}", lowBound=0)
            model += carry_penalty == bank_potential_penalty[w] * (carry_reduction_factor ** i)

    return bank_potential_penalty, model

def set_objective(model: pl.LpProblem, config: OptimizationConfig, data: pd.DataFrame, 
                  vars: Dict, gameweeks: List[int], bank_potential_penalty: dict):
    decay_rate = add_noise(variable_value=config.objective_settings["decay"], standard_deviation=config.objective_settings["decay_sd"], noise_on=config.sensitivity_settings["magnitude"])
    vc_weight = add_noise(variable_value=config.sub_weights["vc"], standard_deviation=config.sub_weights["vc_sd"], noise_on=config.sensitivity_settings["magnitude"])
    itb_value = add_noise(variable_value=config.objective_settings["itb"], standard_deviation=config.objective_settings["itb_sd"], noise_on=config.sensitivity_settings["magnitude"])
    benchg_weight = add_noise(variable_value=config.sub_weights["gk"], standard_deviation=config.sub_weights["gk_sd"], noise_on=config.sensitivity_settings["magnitude"])
    bench1_weight = add_noise(variable_value=config.sub_weights["sub1"], standard_deviation=config.sub_weights["sub1_sd"], noise_on=config.sensitivity_settings["magnitude"])
    bench2_weight = add_noise(variable_value=config.sub_weights["sub2"], standard_deviation=config.sub_weights["sub2_sd"], noise_on=config.sensitivity_settings["magnitude"])
    bench3_weight = add_noise(variable_value=config.sub_weights["sub3"], standard_deviation=config.sub_weights["sub3_sd"], noise_on=config.sensitivity_settings["magnitude"])
    ft_value = add_noise(variable_value=config.ft_settings["ft_value"], standard_deviation=config.ft_settings["ft_value_sd"], noise_on=config.sensitivity_settings["magnitude"])
    carry_values = [0]
    carry_values.append(add_noise(variable_value=config.ft_settings["first_carry"], standard_deviation=config.ft_settings["first_carry_sd"], noise_on=config.sensitivity_settings["magnitude"]))
    carry_values.append(carry_values[1] + add_noise(variable_value=config.ft_settings["second_carry"], standard_deviation=config.ft_settings["second_carry_sd"], noise_on=config.sensitivity_settings["magnitude"]))
    carry_values.append(carry_values[2] + add_noise(variable_value=config.ft_settings["third_carry"], standard_deviation=config.ft_settings["third_carry_sd"], noise_on=config.sensitivity_settings["magnitude"]))
    carry_values.append(carry_values[3] + add_noise(variable_value=config.ft_settings["fourth_carry"], standard_deviation=config.ft_settings["fourth_carry_sd"], noise_on=config.sensitivity_settings["magnitude"]))


    players = data.index.tolist()
    gw_xp = {w: pl.lpSum(data.loc[p, f'{w}_Pts'] * (
        benchg_weight * vars['squad'][p][w] +
        (1 - benchg_weight) * vars['lineup'][p][w] +
        (bench1_weight - benchg_weight) * vars['bench1'][p][w] +
        (bench2_weight - benchg_weight) * vars['bench2'][p][w] +
        (bench3_weight - benchg_weight) * vars['bench3'][p][w] +
        (1 + vars['use_tc'][w]) * vars['captain'][p][w] +
        vc_weight * vars['vicecap'][p][w]) for p in players) for w in gameweeks}

    transfer_costs = {w: pl.lpSum(data.loc[p, 'tv_loss_sell_penalty'] * vars['transfer_out'][p][w] for p in players) for w in gameweeks}
    price_drop_costs = {w: pl.lpSum(data.loc[p, 'hold_penalty'] * vars['squad'][p][w] for p in players) for w in gameweeks}

    number_of_transfers = {w: pl.lpSum(vars['transfer_in'][p][w] for p in players) for w in gameweeks}
    
    penalty_term = {}
    M = 1000
    for w in gameweeks:
        penalty_term[w] = pl.LpVariable(f"penalty_term_{w}", lowBound=0)
        model += penalty_term[w] <= M * vars['transfer_indicator'][w]
        model += penalty_term[w] <= bank_potential_penalty[w]
        model += penalty_term[w] >= bank_potential_penalty[w] - M * (1 - vars['transfer_indicator'][w])
        model += penalty_term[w] >= 0
    

    gw_total = {w: gw_xp[w] - transfer_costs[w] - (4 - ft_value) * vars['hits'][w] + itb_value * vars['in_the_bank'][w] - 
                ft_value * number_of_transfers[w] * (1 - vars['use_wc'][w]) + 
                (1 - vars['use_wc'][w+1]) * pl.lpSum(carry_values[i] * vars['carry_bin'][w, i] for i in range(5)) - penalty_term[w] - price_drop_costs[w]
                for w in gameweeks}

    xp_total = pl.lpSum(gw_total[w] for w in gameweeks)
    xp_decayed = pl.lpSum(gw_total[w] * pow(decay_rate, w-gameweeks[0]) for w in gameweeks)
    model += xp_decayed

    return xp_total, xp_decayed, gw_total, gw_xp

def solve_model(model: pl.LpProblem):
    highs_solver = pl.HiGHS_CMD(
        path='/opt/homebrew/bin/highs',
        timeLimit=None,
        msg=True
    )
    cbc_solver = pl.COIN_CMD(
        path='/opt/homebrew/bin/cbc',
        timeLimit=None,
        msg=True
    )
    status = model.solve(highs_solver)
    print(f"Solver Status: {pl.LpStatus[status]}")
    if status != pl.LpStatusOptimal:
        print("Warning: Solver did not find an optimal solution!")
    return model, status

def write_output(model: pl.LpProblem, config: OptimizationConfig, data: pd.DataFrame, 
                 vars: Dict, gameweeks: List[int], xp_total: pl.LpAffineExpression, xp_decayed: pl.LpAffineExpression, gw_total: pl.LpAffineExpression, gw_xp: pl.LpAffineExpression):

    ftpath = open(f'output/{config.output_options["label"]}-ftpath.txt', 'a')
    ftplayer = open(f'output/{config.output_options["label"]}-ftplayer.txt', 'a')
    xp = open(f'output/{config.output_options["label"]}-xp.txt', 'a')
    lineup = open(f'output/{config.output_options["label"]}-lineup.txt', 'a')    
    player = open(f'output/{config.output_options["label"]}-player.txt', 'a')
    squad = open(f'output/{config.output_options["label"]}-squad.txt', 'a')
    ftpath2 = open(f'output/{config.output_options["label"]}-ftpath2.txt', 'a')
    
    for w in gameweeks:
        squad.write(f'{w}Squad')
        for pos in ['goalkeepers', 'defenders', 'midfielders', 'forwards']:
            for p in data[data['Pos'] == pos[0].upper()].index:
                if vars['squad'][p][w].varValue >= 0.5 and p not in config.force_next_gw["lock"]:
                    squad.write(f' :' + data.loc[p, 'Name'])
        squad.write("\n")
        
        for pos in ['GK', 'Def', 'Mid', 'Fwd']:
            squad.write(f'GW{w}{pos}')
            for p in data[data['Pos'] == pos[0]].index:
                if vars['squad'][p][w].varValue >= 0.5:
                    player.write(f'GW{w}{pos}: ' + data.loc[p, 'Name'] + "\n")
                if vars['squad'][p][w].varValue >= 0.5 and p not in config.force_next_gw["lock"]:
                    squad.write(f' :' + data.loc[p, 'Name'])
            squad.write("\n")
        transfers_used = sum((vars['transfer_in'][p][w].varValue) for p in data.index)
        squad_value = sum((data['SV'][p] * vars['squad'][p][w].varValue) for p in data.index)
        squad.write(f'{w}TV:' + "{:.1f}".format(squad_value) + "\n")
        squad.write(f'{w}ITB:' + "{:.1f}".format(vars['in_the_bank'][w].varValue) + "\n")
        squad.write(f'{w}Used:' + "{:.0f}".format(transfers_used) + "\n")
        squad.write(f'{w}Carry:' + "{:.0f}".format(vars['carry'][w].varValue) + "\n")
        squad.write(f'{w}Hits:' + "{:.0f}".format(vars['hits'][w].varValue) + "\n")

    for w in gameweeks:
        ftpath.write(f'{w}In,')
        for p in data.index:
            if vars['transfer_in'][p][w].varValue >= 0.5:
                ftpath.write(":" + data.loc[p, 'Name'])
                ftplayer.write(f"{w}In:" + data.loc[p, 'Name'] + "\n")        
        ftpath.write(f',{w}Out,')
        for p in data.index:
            if vars['transfer_out'][p][w].varValue >= 0.5:
                ftpath.write(":" + data.loc[p, 'Name'])
                ftplayer.write(f"{w}Out:" + data.loc[p, 'Name'] + "\n")
        ftpath.write("\n")
        ftpath2.write(f'{w}In,')
        for p in data.index:
            if vars['transfer_in'][p][w].varValue >= 0.5:
                ftpath2.write(":" + data.loc[p, 'Name'])
        ftpath2.write("\n" + f'{w}Out,')
        for p in data.index:
            if vars['transfer_out'][p][w].varValue >= 0.5:
                ftpath2.write(":" + data.loc[p, 'Name'])
        ftpath2.write("\n")        
        for p in data.index:
            if vars['captain'][p][w].varValue >= 0.5:
                lineup.write(f'{w}:Captain:' + data.loc[p, 'Name'] + "\n")
        for p in data.index:
            if vars['lineup'][p][w].varValue >= 0.5:
                lineup.write(f'{w}:Lineup:' + data.loc[p, 'Name'] + "\n")

    if config.output_options["decay"] and not config.output_options["nodecay"]:
        xp.write(f"{pl.value(xp_decayed)}" + "\n")
    if config.output_options["nodecay"] and not config.output_options["decay"]:
        xp.write(f"{pl.value(xp_total)}" + "\n")
    if config.output_options["nodecay"] and config.output_options["decay"]:
        xp.write("decay: " + f"{pl.value(xp_decayed)}" + "\n")
        xp.write("nodecay: " + f"{pl.value(xp_total)}" + "\n")
    if config.output_options["gw_points"]:
        for w in gameweeks:
            xp.write(f"{w}:{pl.value(gw_xp[w])}\n")

    ftpath.close()
    ftplayer.close()
    xp.close()
    lineup.close()
    player.close()
    squad.close()

    if config.output_options["composite"]:
        ftpath = open(f'output/{config.output_options["composite_label"]}-ftpath.txt', 'a')
        ftplayer = open(f'output/{config.output_options["composite_label"]}-ftplayer.txt', 'a')
        xp = open(f'output/{config.output_options["composite_label"]}-xp.txt', 'a')
        lineup = open(f'output/{config.output_options["composite_label"]}-lineup.txt', 'a')    
        player = open(f'output/{config.output_options["composite_label"]}-player.txt', 'a')
        squad = open(f'output/{config.output_options["composite_label"]}-squad.txt', 'a')
        ftpath2 = open(f'output/{config.output_options["composite_label"]}-ftpath2.txt', 'a')

        for w in gameweeks:
            squad.write(f'{w}Squad')
            for pos in ['goalkeepers', 'defenders', 'midfielders', 'forwards']:
                for p in data[data['Pos'] == pos[0].upper()].index:
                    if vars['squad'][p][w].varValue >= 0.5 and p not in config.force_next_gw["lock"]:
                        squad.write(f' :' + data.loc[p, 'Name'])
            squad.write("\n")
            
            for pos in ['GK', 'Def', 'Mid', 'Fwd']:
                squad.write(f'GW{w}{pos}')
                for p in data[data['Pos'] == pos[0]].index:
                    if vars['squad'][p][w].varValue >= 0.5:
                        player.write(f'GW{w}{pos}: ' + data.loc[p, 'Name'] + "\n")
                    if vars['squad'][p][w].varValue >= 0.5 and p not in config.force_next_gw["lock"]:
                        squad.write(f' :' + data.loc[p, 'Name'])
                squad.write("\n")
            
            transfers_used = sum((vars['transfer_in'][p][w].varValue) for p in data.index)
            squad_value = sum((data['SV'][p] * vars['squad'][p][w].varValue) for p in data.index)
            squad.write(f'{w}TV:' + "{:.1f}".format(squad_value) + "\n")
            squad.write(f'{w}ITB:' + "{:.1f}".format(vars['in_the_bank'][w].varValue) + "\n")
            squad.write(f'{w}Used:' + "{:.0f}".format(transfers_used) + "\n")
            squad.write(f'{w}Carry:' + "{:.0f}".format(vars['carry'][w].varValue) + "\n")
            squad.write(f'{w}Hits:' + "{:.0f}".format(vars['hits'][w].varValue) + "\n")


        for w in gameweeks:
            ftpath.write(f'{w}In,')
            for p in data.index:
                if vars['transfer_in'][p][w].varValue >= 0.5:
                    ftpath.write(":" + data.loc[p, 'Name'])
                    ftplayer.write(f"{w}In:" + data.loc[p, 'Name'] + "\n")        
            ftpath.write(f',{w}Out,')
            for p in data.index:
                if vars['transfer_out'][p][w].varValue >= 0.5:
                    ftpath.write(":" + data.loc[p, 'Name'])
                    ftplayer.write(f"{w}Out:" + data.loc[p, 'Name'] + "\n")
            ftpath.write("\n")
            ftpath2.write(f'{w}In,')
            for p in data.index:
                if vars['transfer_in'][p][w].varValue >= 0.5:
                    ftpath2.write(":" + data.loc[p, 'Name'])
            ftpath2.write("\n" + f'{w}Out,')
            for p in data.index:
                if vars['transfer_out'][p][w].varValue >= 0.5:
                    ftpath2.write(":" + data.loc[p, 'Name'])
            ftpath2.write("\n")        
            for p in data.index:
                if vars['captain'][p][w].varValue >= 0.5:
                    lineup.write(f'{w}:Captain:' + data.loc[p, 'Name'] + "\n")
            for p in data.index:
                if vars['lineup'][p][w].varValue >= 0.5:
                    lineup.write(f'{w}:Lineup:' + data.loc[p, 'Name'] + "\n")

        if config.output_options["decay"] and not config.output_options["nodecay"]:
            xp.write(f"{config.output_options['label']}:{pl.value(xp_decayed)}\n")
        if config.output_options["nodecay"] and not config.output_options["decay"]:
            xp.write(f"{config.output_options['label']}:{pl.value(xp_total)}\n")
        if config.output_options["nodecay"] and config.output_options["decay"]:
            xp.write(f"{config.output_options['label']}:decay:{pl.value(xp_decayed)}\n")
            xp.write(f"{config.output_options['label']}:nodecay:{pl.value(xp_total)}\n")


        ftpath.close()
        ftplayer.close()
        xp.close()
        lineup.close()
        player.close()
        squad.close()

def run_solver(config: OptimizationConfig):
    data = load_data(config)
    player_lists = create_player_lists(data)
    model, next_gw, gameweeks, all_gw, data = initialize_model(config, data, player_lists)
    data.to_csv('inspect.csv')
    vars = create_decision_variables(player_lists['players'], gameweeks, all_gw)
    # Calculate bank_potential_penalty
    bank_potential_penalty, model = calculate_bank_potential_penalty(vars, gameweeks, config, model)
    add_constraints(model, config, data, player_lists, vars, bank_potential_penalty)
    xp_total, xp_decayed, gw_total, gw_xp = set_objective(model, config, data, vars, gameweeks, bank_potential_penalty)
    solved_model, status = solve_model(model)
    write_output(solved_model, config, data, vars, gameweeks, xp_total, xp_decayed, gw_total, gw_xp)

def main():
    config = OptimizationConfig(
        data_input='11',
        ft_settings={
            "ft_value": 1.6,
            "ft_value_sd": 0.16,
            "first_carry": 0.2,
            "first_carry_sd": 0.02,
            "second_carry": 0.14,
            "second_carry_sd": 0.014,
            "third_carry": 0.05,
            "third_carry_sd": 0.005,
            "fourth_carry": 0.02,
            "fourth_carry_sd": 0.002
        },
        sub_weights={
            "gk": 0.03,
            "gk_sd": 0.006,
            "sub1": 0.3,
            "sub1_sd": 0.06,
            "sub2": 0.1,
            "sub2_sd": 0.02,
            "sub3": 0.01,
            "sub3_sd": 0.002,
            "vc": 0.05,
            "vc_sd": 0.01
        },
        exact_money_penalty={
            "0.0": 0.2,
            "0.0_sd": 0.02,
            "0.1": 0.1,
            "0.1_sd": 0.01,
            "0.2": 0.05,
            "0.2_sd": 0.005,
            "carry_reduction": 0.7,
            "carry_reduction_sd": 0.07
        },
        
        # Team
        transfer_position={"ft_available": 4, "money_itb": 0}, # Simon 
        initial_squad=[3, 15, 17, 120, 129, 99, 372, 252, 291, 311, 328, 366, 82, 495, 536], # Simon,
        # Solver 
        sensitivity_settings={"magnitude": 1, "iterations": 100},
        horizon_settings={"horizon": 11, "future_week": 1, "compress": 0, "compress_sum": 0.15, "csv_weeks": 12},
        objective_settings={"decay": 0.88, "decay_sd": 0.023, "itb": 0.05, "itb_sd": 0.005},
        # EV
        change_ev=[
            [58, -3.6], #Watkins 2nd Pen Taker
            [199, -3], # Eze 2nd Pen Taker
            [207, 3.5], # Mateta 1st Pen Taker
            [447, 1.6], # Wood 1st Pen Taker
            [366, 2.8], # Bruno 80% -> 100%
            [385, -1.8] # Rashford 20% -> 2nd Taker
            ],
        reduce_availability=[],
        undecay_ev=False,
        # Transfer
        force_next_gw={"ban": [], "lock": []},
        force_horizon={"ban": [172], "lock": []},
        force_particular_gw={"ban": [], "lock": []},
        priority=[],
        priority_plus=[],
        soft_winner=[],
        force_roll=[],
        # Sell Penalty
        tied_up_value=0,
        buyback_likelihood = [[291, 0], [15, 0.5], [3, 0.5], [82, 0.5], [99, 0.75], [328, 0.5], [252, 0], [17, 0.5], [372, 0]], # Simon
        price_drop_rate=[],
        chip_weeks={'wc': 19, 'bb': 0, 'fh': 0, 'tc': 0},
        # Output        
        output_options={"label": "12-0",
                       "composite": False,
                       "composite_label": "wc",
                       "decay": True, 
                       "nodecay": False,
                       "gw_points": False},
    )
    for _ in range(config.sensitivity_settings["iterations"]):
        run_solver(config)

if __name__ == "__main__":
    main()

