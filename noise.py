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

# Create Lists By Position
data = df.copy().reset_index()
data.set_index('ID', inplace=True)
players = data.index.tolist()

data_g = df[df['Pos']== 'G'].copy().reset_index()
data_g.set_index('ID', inplace=True)
goalkeepers = data_g.index.tolist()

data_d = df[df['Pos']== 'D'].copy().reset_index()
data_d.set_index('ID', inplace=True)
defenders = data_d.index.tolist()

data_m = df[df['Pos']== 'M'].copy().reset_index()
data_m.set_index('ID', inplace=True)
midfielders = data_m.index.tolist()

data_f = df[df['Pos']== 'F'].copy().reset_index()
data_f.set_index('ID', inplace=True)
forwards = data_f.index.tolist()

# Create Lists By Team
data_ars = df[df['Team'] == 'Arsenal'].copy().reset_index()
data_ars.set_index('ID', inplace=True)
arsenal = data_ars.index.tolist()

data_avl = df[df['Team'] == 'Aston Villa'].copy().reset_index()
data_avl.set_index('ID', inplace=True)
aston_villa = data_avl.index.tolist()

data_bha = df[df['Team'] == 'Brighton'].copy().reset_index()
data_bha.set_index('ID', inplace=True)
brighton = data_bha.index.tolist()

data_bur = df[df['Team'] == 'Burnley'].copy().reset_index()
data_bur.set_index('ID', inplace=True)
burnley = data_bur.index.tolist()

data_bre = df[df['Team'] == 'Brentford'].copy().reset_index()
data_bre.set_index('ID', inplace=True)
brentford = data_bre.index.tolist()

data_che = df[df['Team'] == 'Chelsea'].copy().reset_index()
data_che.set_index('ID', inplace=True)
chelsea = data_che.index.tolist()

data_cpl = df[df['Team'] == 'Crystal Palace'].copy().reset_index()
data_cpl.set_index('ID', inplace=True)
crystal_palace = data_cpl.index.tolist()

data_eve = df[df['Team'] == 'Everton'].copy().reset_index()
data_eve.set_index('ID', inplace=True)
everton = data_eve.index.tolist()

data_lee = df[df['Team'] == 'Leeds'].copy().reset_index()
data_lee.set_index('ID', inplace=True)
leeds = data_lee.index.tolist()

data_lei = df[df['Team'] == 'Leicester'].copy().reset_index()
data_lei.set_index('ID', inplace=True)
leicester = data_lee.index.tolist()

data_liv = df[df['Team'] == 'Liverpool'].copy().reset_index()
data_liv.set_index('ID', inplace=True)
liverpool = data_liv.index.tolist()

data_mci = df[df['Team'] == 'Man City'].copy().reset_index()
data_mci.set_index('ID', inplace=True)
man_city = data_mci.index.tolist()

data_mun = df[df['Team'] == 'Man Utd'].copy().reset_index()
data_mun.set_index('ID', inplace=True)
man_utd = data_mun.index.tolist()

data_new = df[df['Team'] == 'Newcastle'].copy().reset_index()
data_new.set_index('ID', inplace=True)
newcastle = data_new.index.tolist()

data_nor = df[df['Team'] == 'Norwich'].copy().reset_index()
data_nor.set_index('ID', inplace=True)
norwich = data_nor.index.tolist()

data_sou = df[df['Team'] == 'Southampton'].copy().reset_index()
data_sou.set_index('ID', inplace=True)
southampton = data_sou.index.tolist()

data_tot = df[df['Team'] == 'Spurs'].copy().reset_index()
data_tot.set_index('ID', inplace=True)
tottenham = data_tot.index.tolist()

data_wat = df[df['Team'] == 'Watford'].copy().reset_index()
data_wat.set_index('ID', inplace=True)
watford = data_wat.index.tolist()

data_whu = df[df['Team'] == 'West Ham'].copy().reset_index()
data_whu.set_index('ID', inplace=True)
west_ham = data_whu.index.tolist()

data_wol = df[df['Team'] == 'Wolves'].copy().reset_index()
data_wol.set_index('ID', inplace=True)
wolves = data_wol.index.tolist()

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