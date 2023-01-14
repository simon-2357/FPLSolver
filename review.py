import pandas as pd
import pulp as pl
import numpy as np

df = pd.read_csv('data/20-deadline-odds25.csv')
df.set_index('ID', inplace=True)
data = df.copy().reset_index()
data.set_index('ID', inplace=True)
players = data.index.tolist()

# Create Lists of Player IDs by Position
pos_group = data.groupby(data['Pos'])
goalkeepers = (pos_group.get_group('G')).index.tolist()
defenders = (pos_group.get_group('D')).index.tolist()
midfielders = (pos_group.get_group('M')).index.tolist()
forwards = (pos_group.get_group('F')).index.tolist()

# Create Lists of Player IDs by Team
team_group = data.groupby(data['Team'])
arsenal = (team_group.get_group('Arsenal')).index.tolist()
aston_villa = (team_group.get_group('Aston Villa')).index.tolist()
brighton = (team_group.get_group('Brighton')).index.tolist()
fulham = (team_group.get_group('Fulham')).index.tolist()
brentford = (team_group.get_group('Brentford')).index.tolist()
chelsea = (team_group.get_group('Chelsea')).index.tolist()
crystal_palace = (team_group.get_group('Crystal Palace')).index.tolist()
everton = (team_group.get_group('Everton')).index.tolist()
leeds = (team_group.get_group('Leeds')).index.tolist()
leicester = (team_group.get_group('Leicester')).index.tolist()
liverpool = (team_group.get_group('Liverpool')).index.tolist()
man_city = (team_group.get_group('Man City')).index.tolist()
man_utd = (team_group.get_group('Man Utd')).index.tolist()
newcastle = (team_group.get_group('Newcastle')).index.tolist()
bournemouth = (team_group.get_group('Bournemouth')).index.tolist()
southampton = (team_group.get_group('Southampton')).index.tolist()
tottenham = (team_group.get_group('Spurs')).index.tolist()
forest = (team_group.get_group("Nott'm Forest")).index.tolist()
west_ham = (team_group.get_group('West Ham')).index.tolist()
wolves = (team_group.get_group('Wolves')).index.tolist()


def run_solver():
    # Options
    bb_week = 39
    wc_week = 39
    tc_week = 39
    fh_week = 39
    bank = 3
    ft_input = 2
    initial_squad = [307, 112, 16, 357, 346, 19, 13, 283, 335, 210, 318, 133, 332, 237, 146]

    # initial_squad =
    decay_rate = 0.85
    vc_weight = 0.05
    horizon = 9
    noise_magnitude = 1
    no_transfer_weeks = []
    banned_players = []
    essential_players = []

    if horizon == 1:
        ft_value = 0
        itb_value = 0
        benchg_weight = 0.02
        bench1_weight = 0.2
        bench2_weight = 0.04
        bench3_weight = 0
        burn_value = 0
    else:
        ft_value = 1
        two_ft_value = 1.5
        itb_value = 0.1
        benchg_weight = 0.02
        bench1_weight = 0.2
        bench2_weight = 0.04
        bench3_weight = 0

    data = df.copy().reset_index()
    data.set_index('ID', inplace=True)

    # Model Set up
    model = pl.LpProblem('model', pl.LpMaximize)
    solver = pl.PULP_CBC_CMD()

    # Find Next GW & Generate GW List
    next_gw = int(df.keys()[6].split('_')[0])
    gameweeks = list(range(next_gw, next_gw + horizon))
    all_gw = [next_gw - 1] + gameweeks
    gwminus = list(range(next_gw, next_gw + horizon - 1))

    if fh_week in gameweeks:
        gameweeks = gwminus
        horizon = horizon - 1
        if fh_week < wc_week:
            wc_week = wc_week - 1
        if fh_week < bb_week:
            bb_week = bb_week - 1
        for w in gwminus:
            if w >= fh_week:
                data[f'{w}_Pts'] = data[f'{w + 1}_Pts']

    # Decision Variables
    lineup = pl.LpVariable.dicts('lineup', (players, gameweeks), 0, 1, cat='Integer')
    squad = pl.LpVariable.dicts('squad', (players, all_gw), 0, 1, cat='Integer')
    bench1 = pl.LpVariable.dicts('bench1', (players, gameweeks), 0, 1, cat='Integer')
    bench2 = pl.LpVariable.dicts('bench2', (players, gameweeks), 0, 1, cat='Integer')
    bench3 = pl.LpVariable.dicts('bench3', (players, gameweeks), 0, 1, cat='Integer')
    captain = pl.LpVariable.dicts('captain', (players, gameweeks), 0, 1, cat='Integer')
    vicecap = pl.LpVariable.dicts('vicecap', (players, gameweeks), 0, 1, cat='Integer')
    transfer_in = pl.LpVariable.dicts('transfer_in', (players, gameweeks), 0, 1, cat='Integer')
    transfer_out = pl.LpVariable.dicts('transfer_out', (players, gameweeks), 0, 1, cat='Integer')
    in_the_bank = pl.LpVariable.dicts('itb', all_gw, 0)
    free_transfers = pl.LpVariable.dicts('ft', all_gw, 1, 15, cat='Integer')
    hits = pl.LpVariable.dicts('hits', gameweeks, 0, cat='Integer')
    carry = pl.LpVariable.dicts('carry', all_gw, 0, 1, cat='Integer')
    use_bb = pl.LpVariable.dicts('use_bb', gameweeks, 0, 1, cat='Integer')
    use_wc = pl.LpVariable.dicts('use_wc', gameweeks, 0, 1, cat='Integer')
    use_tc = pl.LpVariable.dicts('use_tc', gameweeks, 0, 1, cat='Integer')

    rng = np.random.default_rng()
    data.loc[data['Pos'] == 'G', ['pos_noise']] = -0.0176
    data.loc[data['Pos'] == 'D', ['pos_noise']] = 0
    data.loc[data['Pos'] == 'M', ['pos_noise']] = -0.0553
    data.loc[data['Pos'] == 'F', ['pos_noise']] = -0.0414
    for w in gameweeks:
        noise = (0.7293 + data[f'{w}_Pts'] * 0.0044 - data[f'{w}_xMins'] * 0.0083 + (w - next_gw) * 0.0092 + data['pos_noise']) * rng.standard_normal(size=len(players)) * noise_magnitude
        data[f'{w}_Pts'] = data[f'{w}_Pts'] * (1 + noise)

    # Budget Things
    player_sv = data['SV'].to_dict()
    player_bv = data['BV'].to_dict()
    sold_amount = {w: pl.lpSum(player_sv[p] * transfer_out[p][w] for p in players) for w in gameweeks}
    bought_amount = {w: pl.lpSum(player_bv[p] * transfer_in[p][w] for p in players) for w in gameweeks}
    points_player_week = {p: {w: data.loc[p, f'{w}_Pts'] for w in gameweeks} for p in players}
    number_of_transfers = {w: pl.lpSum(0.5 * (transfer_in[p][w] + transfer_out[p][w]) for p in players) for w in gameweeks}
    carry[next_gw-1] = ft_input - 1

    # Don't roll transfer before a WC/FH, can't roll transfer out of a WC/FH
    if wc_week in gameweeks:
        model += carry[wc_week-1] <= 0
        model += carry[wc_week] <= 0

    if fh_week in gameweeks:
        model += carry[fh_week-1] <= 0

    # Set Initial Conditions
    for p in players:
        if p in initial_squad:
            squad[p][next_gw-1] = 1
        else:
            squad[p][next_gw-1] = 0

    in_the_bank[next_gw-1] = bank

    # Import use of chips into the model
    for w in gameweeks:
        if w == bb_week:
            use_bb[w] = 1
        else:
            use_bb[w] = 0
        if w == wc_week:
            use_wc[w] = 1
        else:
            use_wc[w] = 0
        if w == tc_week:
            use_tc[w] = 1
        else:
            use_tc[w] = 0

    # Import banned and essential players into the model
    for p in banned_players:
        model += squad[p][next_gw] == 0
    for p in essential_players:
        model += squad[p][next_gw] == 1

    # Objective Variable
    gw_xp = {w: pl.lpSum(points_player_week[p][w] * (benchg_weight * squad[p][w] + (1 - benchg_weight) * lineup[p][w] + (bench1_weight - benchg_weight) * bench1[p][w] + (bench2_weight - benchg_weight) * bench2[p][w] + (bench3_weight - benchg_weight) * bench3[p][w] + (1 + use_tc[w]) * captain[p][w] + vc_weight * vicecap[p][w]) for p in players) for w in gameweeks}
    gw_total = {w: gw_xp[w] - 3 * hits[w] + itb_value * in_the_bank[w] - ft_value * number_of_transfers[w] * (1 - use_wc[w]) + two_ft_value * carry[w] for w in gameweeks}
    model += pl.lpSum(gw_total[w] * pow(decay_rate, w-next_gw) for w in gameweeks)

    # Squad Mechanics
    for w in gameweeks:
        model += in_the_bank[w] - in_the_bank[w-1] == sold_amount[w] - bought_amount[w]
        model += in_the_bank[w] >= 0
        model += free_transfers[w] == carry[w-1] + 1 + 14 * use_wc[w]
        model += free_transfers[w] - number_of_transfers[w] + hits[w] >= carry[w]
        model += carry[w] <= 1
        model += hits[w] >= number_of_transfers[w] - free_transfers[w]
        for p in players:
            model += squad[p][w] - squad[p][w - 1] == transfer_in[p][w] - transfer_out[p][w]

    for w in no_transfer_weeks:
        free_transfers[w] = 0
        carry[w] == 0
        model += hits[w] == number_of_transfers[w]

    # Valid Squad Formation
    for w in gameweeks:
        model += pl.lpSum(squad[p][w] for p in players) == 15
        model += pl.lpSum(squad[g][w] for g in goalkeepers) == 2
        model += pl.lpSum(squad[d][w] for d in defenders) == 5
        model += pl.lpSum(squad[m][w] for m in midfielders) == 5
        model += pl.lpSum(squad[f][w] for f in forwards) == 3
        model += pl.lpSum(lineup[p][w] for p in players) == 11 + 4 * use_bb[w]
        model += pl.lpSum(bench1[p][w] for p in players) == 1 - use_bb[w]
        model += pl.lpSum(bench2[p][w] for p in players) == 1 - use_bb[w]
        model += pl.lpSum(bench3[p][w] for p in players) == 1 - use_bb[w]
        model += pl.lpSum(bench1[g][w] for g in goalkeepers) == 0
        model += pl.lpSum(bench2[g][w] for g in goalkeepers) == 0
        model += pl.lpSum(bench3[g][w] for g in goalkeepers) == 0
        model += pl.lpSum(lineup[g][w] for g in goalkeepers) == 1 + use_bb[w]
        model += pl.lpSum(lineup[d][w] for d in defenders) >= 3
        model += pl.lpSum(lineup[d][w] for d in defenders) <= 5
        model += pl.lpSum(lineup[m][w] for m in midfielders) >= 2
        model += pl.lpSum(lineup[m][w] for m in midfielders) <= 5
        model += pl.lpSum(lineup[f][w] for f in forwards) >= 1
        model += pl.lpSum(lineup[f][w] for f in forwards) <= 3
        model += pl.lpSum(captain[p][w] for p in players) == 1
        model += pl.lpSum(vicecap[p][w] for p in players) == 1
        model += pl.lpSum(squad[x][w] for x in arsenal) <= 3
        model += pl.lpSum(squad[x][w] for x in aston_villa) <= 3
        model += pl.lpSum(squad[x][w] for x in brentford) <= 3
        model += pl.lpSum(squad[x][w] for x in brighton) <= 3
        model += pl.lpSum(squad[x][w] for x in forest) <= 3
        model += pl.lpSum(squad[x][w] for x in chelsea) <= 3
        model += pl.lpSum(squad[x][w] for x in crystal_palace) <= 3
        model += pl.lpSum(squad[x][w] for x in everton) <= 3
        model += pl.lpSum(squad[x][w] for x in leeds) <= 3
        model += pl.lpSum(squad[x][w] for x in leicester) <= 3
        model += pl.lpSum(squad[x][w] for x in liverpool) <= 3
        model += pl.lpSum(squad[x][w] for x in man_city) <= 3
        model += pl.lpSum(squad[x][w] for x in man_utd) <= 3
        model += pl.lpSum(squad[x][w] for x in newcastle) <= 3
        model += pl.lpSum(squad[x][w] for x in bournemouth) <= 3
        model += pl.lpSum(squad[x][w] for x in southampton) <= 3
        model += pl.lpSum(squad[x][w] for x in tottenham) <= 3
        model += pl.lpSum(squad[x][w] for x in fulham) <= 3
        model += pl.lpSum(squad[x][w] for x in west_ham) <= 3
        model += pl.lpSum(squad[x][w] for x in wolves) <= 3

    # Lineup & Cap Within Squad, C/VC Different
    for w in gameweeks:
        for p in players:
            model += lineup[p][w] <= squad[p][w]
            model += captain[p][w] <= lineup[p][w]
            model += vicecap[p][w] <= lineup[p][w]
            model += vicecap[p][w] + captain[p][w] <= 1
            model += bench1[p][w] <= squad[p][w]
            model += bench2[p][w] <= squad[p][w]
            model += bench3[p][w] <= squad[p][w]
            model += bench1[p][w] + bench2[p][w] + bench3[p][w] <= 1
    model.solve(solver)
    f = open('deadline1.txt', 'a')
    g = open('deadline2.txt', 'a')
    # for p in players:
    #     if captain[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw}:Captain:' + data['Name'][p] + "\n")
    # for p in players:
    #     if lineup[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw}:Lineup:' + data['Name'][p] + "\n")
    # for p in goalkeepers:
    #     if squad[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw} GK: ' + data['Name'][p] + "\n")
    # for p in defenders:
    #     if squad[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw} Def: ' + data['Name'][p] + "\n")
    # for p in midfielders:
    #     if squad[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw} Mid: ' + data['Name'][p] + "\n")
    # for p in forwards:
    #     if squad[p][next_gw].varValue >= 0.5:
    #         f.write(f'{next_gw} Fwd: ' + data['Name'][p] + "\n")
    for w in gameweeks:
        f.write(f'{w}In,')
        for p in players:
            if transfer_in[p][w].varValue >= 0.5:
                f.write(":" + data['Name'][p])
                g.write(f"{w}In:" + data['Name'][p] + "\n")
    
        f.write(f',{w}Out,')
        for p in players:
            if transfer_out[p][w].varValue >= 0.5:
                f.write(":" + data['Name'][p])
                g.write(f"{w}Out:" + data['Name'][p] + "\n")
    # #     # f.write("\n"+ f"{w}Hits:{hits[w].varValue}" + "\n")
        f.write("\n")



# def print_transfers():
#     for w in gameweeks:
#         if w > wc_week:
#             for p in players:
#                 if transfer_in[p][w].varValue >= 0.5:
#                     print(f'{w} In: ' + data['Name'][p])
#
#             for p in players:
#                 if transfer_out[p][w].varValue >= 0.5:
#                     print(f'{w} Out: ' + data['Name'][p])

# def print_lineup(w):
#          for p in goalkeepers:
#              if lineup[p][w].varValue >= 0.5:
#               print(f'{w} GK: ' + data['Name'][p])
#          for p in defenders:
#              if lineup[p][w].varValue >= 0.5:
#                  print(f'{w} Def: ' + data['Name'][p])
#          for p in midfielders:
#              if lineup[p][w].varValue >= 0.5:
#                  print(f'{w} Mid: ' + data['Name'][p])
#          for p in forwards:
#              if lineup[p][w].varValue >= 0.5:
#                  print(f'{w} Fwd: ' + data['Name'][p])


# def print_squad(w):
#     for p in goalkeepers:
#         if squad[p][w].varValue >= 0.5:
#             print(f'{w} GK: ' + data['Name'][p])
#     for p in defenders:
#         if squad[p][w].varValue >= 0.5:
#             print(f'{w} Def: ' + data['Name'][p])
#     for p in midfielders:
#         if squad[p][w].varValue >= 0.5:
#             print(f'{w} Mid: ' + data['Name'][p])
#     for p in forwards:
#         if squad[p][w].varValue >= 0.5:
#             print(f'{w} Fwd: ' + data['Name'][p])

# def print_captain(w):
#     for p in players:
#         if captain[p][w].varValue >= 0.5:
#             print(f'{w} Captain: ' + data['Name'][p])


# def squad_to_txt(w):
#     f = open('36-no3536.txt', 'a')
#     for p in goalkeepers:
#         if squad[p][w].varValue >= 0.5:
#             f.write(f'{w} GK: ' + data['Name'][p] + "\n")
#     for p in defenders:
#         if squad[p][w].varValue >= 0.5:
#             f.write(f'{w} Def: ' + data['Name'][p] + "\n")
#     for p in midfielders:
#         if squad[p][w].varValue >= 0.5:
#             f.write(f'{w} Mid: ' + data['Name'][p] + "\n")
#     for p in forwards:
#         if squad[p][w].varValue >= 0.5:
#             f.write(f'{w} Fwd: ' + data['Name'][p] + "\n")

# def transfers_to_txt():
#     f = open('36-no3536.txt', 'a')
#     for w in gameweeks:
#         if w > wc_week:
#             for p in players:
#                 if transfer_in[p][w].varValue >= 0.5:
#                     f.write(f'{w} In: ' + data['Name'][p] + "\n")
#
#                 if transfer_out[p][w].varValue >= 0.5:
#                     f.write(f'{w} Out: ' + data['Name'][p] + "\n")

solver_runs = 25
for x in range(solver_runs):
    run_solver()
