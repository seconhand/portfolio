# -*- coding: utf-8 -*
#%%
# NBA 팀들의 데이터프레임
from nba_api.stats.static import teams
import pandas as pd
nba_teams = teams.get_teams()
teams_info = []
for i in range(len(nba_teams)):
    teams_info.append(nba_teams[i])
teams_info = pd.DataFrame(teams_info) 
'''
Index(['id', 'full_name', 'abbreviation', 'nickname', 'city', 'state',
       'year_founded'],
      dtype='object')
'''
#%%
#player 데이터프레임
from nba_api.stats.static import players
nba_players = players.get_active_players()
players_info = []
for i in range(len(nba_players)):
    players_info.append(nba_players[i])
players_info = pd.DataFrame(players_info)
'''
print(players_info.columns) 
Index(['id', 'full_name', 'first_name', 'last_name', 'is_active'], dtype='object')
'''
#%%
from nba_api.stats.endpoints import PlayerCareerStats
curry_id = players_info.loc[players_info['full_name']=='Stephen Curry'].id.values
curry = PlayerCareerStats(curry_id)
curry_career = curry.get_data_frames()
#%%
#원하는 선수 기록 가져오기
def total_career(name):
    player_id = players_info.loc[players_info.full_name == name].id.values
    career = PlayerCareerStats(player_id)
    df_career = career.get_data_frames()[0]
    return df_career
#%%
from nba_api.stats.endpoints import TeamDashLineups
gsw_id = teams_info.loc[teams_info.abbreviation == 'GSW'].id.values
gsw_career = TeamDashLineups(gsw_id)  
df_gsw = gsw_career.get_data_frames()

#%%
#원하는 팀 기록 가져오기
def team_info(name):
    




