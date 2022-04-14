import pandas as pd
import pulp as pl
import numpy as np
import urllib
import json

# Manually set FH week, uncomment 157/158 to undo
# next_gw = 37
# gameweeks = [37]

# Chip Settings
bb_week = 36
wc_week = 34
tc_week = 26
fh_week = 33

df = pd.read_csv('data/37353637.csv')

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
burnley = (team_group.get_group('Burnley')).index.tolist()
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
norwich = (team_group.get_group('Norwich')).index.tolist()
southampton = (team_group.get_group('Southampton')).index.tolist()
tottenham = (team_group.get_group('Spurs')).index.tolist()
watford = (team_group.get_group('Watford')).index.tolist()
west_ham = (team_group.get_group('West Ham')).index.tolist()
wolves = (team_group.get_group('Wolves')).index.tolist()

# Model Set up
model = pl.LpProblem('model', pl.LpMaximize)
solver = pl.PULP_CBC_CMD()

# Team Setup
fpl_id = 1049
bank = 4.2
ft_input = 1
initial_squad = [22, 279, 6, 69, 71, 196, 215, 233, 315, 359, 360, 370, 475, 439, 700]
banned_players = []
essential_players = []

# Model Parameters
decay_rate = 1
vc_weight = 0.05
horizon = 6
noise_magnitude = 0
solver_runs = 1

if horizon == 1:
    ft_value = 0
    itb_value = 0
    bench_weight = 0
else:
    ft_value = (1.5 * horizon / 8)
    itb_value = 0.1
    bench_weight = 0.1

# Get initial squad
url = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/event/31/picks/"
response = urllib.request.urlopen(url)
fpl_json = json.loads(response.read())
# sub_dict = fpl_json['items'][0]['participants']

# Arne Initial Squad
# [295, 16, 234, 237, 360, 233, 22, 51, 196, 6, 425, 505, 357, 170, 700] - Arne
# [22, 16, 6, 69, 71, 196, 215, 233, 315, 359, 360, 370, 475, 439, 700] - Me
# [22, 16, 6, 196, 233, 359, 360, 475, 700, 142, 701, 518, 168, 237, 295] - Andy
# [69, 256, 360, 67, 233, 681, 196, 359, 22, 6, 700, 475, 168, 439, 468] - Abdul

# Find Next GW & Generate GW List
next_gw = int(df.keys()[6].split('_')[0])
gameweeks = list(range(next_gw,next_gw+horizon))
all_gw = [next_gw-1] + gameweeks
gwminus = list(range(next_gw,next_gw+horizon-1))


# Setting Up Seeds & Coefficients For Adding Noise
rng = np.random.default_rng()
data.loc[data['Pos'] == 'G', ['pos_noise']] = -0.0176
data.loc[data['Pos'] == 'D', ['pos_noise']] = 0
data.loc[data['Pos'] == 'M', ['pos_noise']] = -0.0553
data.loc[data['Pos'] == 'F', ['pos_noise']] = -0.0414

# Add Noise To Point Data
for w in gameweeks:
       noise = (0.7293 + data[f'{w}_Pts'] * 0.0044 - data[f'{w}_xMins'] * 0.0083 + (w-next_gw)*0.0092 + data['pos_noise']) * rng.standard_normal(size=len(players)) * noise_magnitude
       data[f'{w}_Pts'] = data[f'{w}_Pts'] * (1 + noise)

# Free Hit Logic - Optimise for everything but the Free Hit Week
for w in gwminus:
    if w >= fh_week:
        data[f'{w}_Pts'] = data[f'{w+1}_Pts']

if fh_week in gameweeks:
    gameweeks = gwminus
    horizon = horizon - 1
    if fh_week < wc_week:
         wc_week = wc_week - 1
    if fh_week < bb_week:
        bb_week = bb_week - 1



# Decision Variables
lineup = pl.LpVariable.dicts('lineup', (players, gameweeks), 0, 1, cat='Integer')
squad = pl.LpVariable.dicts('squad', (players, all_gw), 0, 1, cat='Integer')
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

#Budget Things
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

#Set Initial Conditions
for p in players:
    if p in initial_squad:
        squad[p][next_gw-1] = 1
    else:
        squad[p][next_gw-1] = 0

in_the_bank[next_gw-1] = bank

#Import use of chips into the model
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
for w in gameweeks:
    for p in banned_players:
        squad[p][w] = 0
    for p in essential_players:
        squad[p][w] = 1



# Objective Variable
gw_xp = {w: pl.lpSum(points_player_week[p][w] * (bench_weight * squad[p][w] + (1 - bench_weight) * lineup[p][w] + (1 + use_tc[w]) * captain[p][w] + vc_weight * vicecap[p][w]) for p in players) for w in gameweeks}
gw_total = {w: gw_xp[w] - 4 * hits[w] + itb_value * in_the_bank[w] + ft_value * carry[w] for w in gameweeks}
model += pl.lpSum(gw_total[w] for w in gameweeks)

# Squad Mechanics
for w in gameweeks:
    model += in_the_bank[w] - in_the_bank[w-1] == sold_amount[w] - bought_amount[w]
    model += in_the_bank[w] >= 0
    model += free_transfers[w] == carry[w-1] + 1 + 14 * use_wc[w]
    model += free_transfers[w] - number_of_transfers[w] >= carry[w]
    model += carry[w] <= 1
    model += hits[w] >= number_of_transfers[w] - free_transfers[w]

    for p in players:
        model += squad[p][w] - squad[p][w - 1] == transfer_in[p][w] - transfer_out[p][w]

# Valid Squad Formation
for w in gameweeks:
    model += pl.lpSum(squad[p][w] for p in players) == 15
    model += pl.lpSum(squad[g][w] for g in goalkeepers) == 2
    model += pl.lpSum(squad[d][w] for d in defenders) == 5
    model += pl.lpSum(squad[m][w] for m in midfielders) == 5
    model += pl.lpSum(squad[f][w] for f in forwards) == 3
    model += pl.lpSum(lineup[p][w] for p in players) == 11 + 4 * use_bb[w]
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
    model += pl.lpSum(squad[x][w] for x in burnley) <= 3
    model += pl.lpSum(squad[x][w] for x in chelsea) <= 3
    model += pl.lpSum(squad[x][w] for x in crystal_palace) <= 3
    model += pl.lpSum(squad[x][w] for x in everton) <= 3
    model += pl.lpSum(squad[x][w] for x in leeds) <= 3
    model += pl.lpSum(squad[x][w] for x in leicester) <= 3
    model += pl.lpSum(squad[x][w] for x in liverpool) <= 3
    model += pl.lpSum(squad[x][w] for x in man_city) <= 3
    model += pl.lpSum(squad[x][w] for x in man_utd) <= 3
    model += pl.lpSum(squad[x][w] for x in newcastle) <= 3
    model += pl.lpSum(squad[x][w] for x in norwich) <= 3
    model += pl.lpSum(squad[x][w] for x in southampton) <= 3
    model += pl.lpSum(squad[x][w] for x in tottenham) <= 3
    model += pl.lpSum(squad[x][w] for x in watford) <= 3
    model += pl.lpSum(squad[x][w] for x in west_ham) <= 3
    model += pl.lpSum(squad[x][w] for x in wolves) <= 3

# Lineup & Cap Within Squad, C/VC Different
for w in gameweeks:
    for p in players:
        model += lineup[p][w] <= squad[p][w]
        model += captain[p][w] <= lineup[p][w]
        model += vicecap[p][w] <= lineup[p][w]
        model += vicecap[p][w] + captain[p][w] <= 1

for x in range(solver_runs):
    model.solve(solver)

def print_transfers():
    for w in gameweeks:
        for p in players:
            if transfer_in[p][w].varValue >= 0.5:
                print(f'{w} In: ' + data['Name'][p])

            if transfer_out[p][w].varValue >= 0.5:
                print(f'{w} Out: ' + data['Name'][p])

def print_lineup(w):
         for p in goalkeepers:
             if lineup[p][w].varValue >= 0.5:
              print(f'{w} GK: ' + data['Name'][p])
         for p in defenders:
             if lineup[p][w].varValue >= 0.5:
                 print(f'{w} Def: ' + data['Name'][p])
         for p in midfielders:
             if lineup[p][w].varValue >= 0.5:
                 print(f'{w} Mid: ' + data['Name'][p])
         for p in forwards:
             if lineup[p][w].varValue >= 0.5:
                 print(f'{w} Fwd: ' + data['Name'][p])

def print_squad(w):
            for p in goalkeepers:
                if squad[p][w].varValue >= 0.5:
                    print(f'{w} GK: ' + data['Name'][p])
            for p in defenders:
                 if squad[p][w].varValue >= 0.5:
                      print(f'{w} Def: ' + data['Name'][p])
            for p in midfielders:
                   if squad[p][w].varValue >= 0.5:
                    print(f'{w} Mid: ' + data['Name'][p])
            for p in forwards:
                  if squad[p][w].varValue >= 0.5:
                     print(f'{w} Fwd: ' + data['Name'][p])

print_squad(34)