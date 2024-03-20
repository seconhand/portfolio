import pandas as pd
from nba_api.stats.static import players,teams
from nba_api.stats.endpoints import PlayerCareerStats,TeamGameLogs
from nba_api.stats.endpoints import PlayerProfileV2
#%%
#NBA 팀,선수 정보디테일 데이터프레임화
nba_teams = teams.get_teams()
nba_teams = pd.DataFrame(nba_teams)
nba_players = players.get_players()
nba_players = pd.DataFrame(nba_players)
nba_players = nba_players.loc[nba_players.is_active == True]
nba_players.reset_index(inplace = True) # 현역선수만
#%%
cols = ['PLAYER_ID', 'LEAGUE_ID', 'TEAM_ID', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 
       'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 
       'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']
players_career = pd.DataFrame(columns=cols)
#%%
#선수 통산성적표
# append 보다 concat 이 더 빠름
ids =  nba_players.id.values
names = nba_players.full_name.values
for i in ids:
    career = PlayerProfileV2(player_id = i)
    career = career.get_data_frames()[1]# 통산 성적
    players_career = pd.concat([players_career,career],axis=0,ignore_index=True)
#%%
#players_career['full_name'] = [name ]
#james = PlayerProfileV2(nba_players.id[nba_players.full_name == 'LeBron James']).get_data_frames()[1][0]
#curry = PlayerProfileV2(nba_players.id[nba_players.full_name == 'Stephen Curry']).get_data_frames()[1][0]
#%%
with pd.ExcelWriter('nba_player_career.xlsx') as w:
    players_career.to_excel(w,sheet_name = 'players_career',index = False)
with pd.ExcelWriter('nba_teams.xlsx') as w:
    nba_teams.to_excel(w,sheet_name = 'players',index = False)
with pd.ExcelWriter('nba_player.xlsx') as w:
    nba_players.to_excel(w,sheet_name = 'teams',index = False)
#%%    
#curry 커리어

from nba_api.stats.endpoints import PlayerCareerStats, PlayerCompare
curry_id = nba_players.id[nba_players['full_name']=='Stephen Curry'].values
c_curry = PlayerCareerStats(player_id=curry_id).get_data_frames()
curry_regular_season = c_curry[0]
curry_total_R = c_curry[1]
curry_post_season = c_curry[2]
curry_total_P = c_curry[3]
curry_cnt_allstar = len(c_curry[4])
#======================================
#magic 커리어
magic_id = players.find_players_by_full_name("Magic Johnson")[0]['id']
c_magic = PlayerCareerStats(player_id=magic_id).get_data_frames()
magic_regular_season = c_magic[0]
magic_total_R = c_magic[1]
magic_post_season = c_magic[2]
magic_total_P = c_magic[3]
magic_cnt_allstar = len(c_magic[4])

'''
get data frame
0-시즌별 커리어
1-통산 커리어 기록
2-시즌별 포스트 시즌 기록
3-통산 포스트시즌 기록
4-시즌별 올스타 기록
5-통산 올스타 기록
'''
#%%
with pd.ExcelWriter('curry_ReagularSeason.xlsx') as w:
    curry_regular_season.to_excel(w,sheet_name = 'curry_reason',index = False)
with pd.ExcelWriter('curry.xlsx') as w:
    curry_total_R.to_excel(w,sheet_name = 'curry_r_total',index = False)    
with pd.ExcelWriter('curry_PostSeason.xlsx') as w:
    curry_post_season.to_excel(w,sheet_name = 'curry_p_season',index = False)
with pd.ExcelWriter('curry_p_total.xlsx') as w:
    curry_total_P.to_excel(w,sheet_name = 'curry_p_total',index = False)
#=================================================
with pd.ExcelWriter('magic_ReagularSeason.xlsx') as w:
    magic_regular_season.to_excel(w,sheet_name = 'magic_reason',index = False)
with pd.ExcelWriter('magic.xlsx') as w:
    magic_total_R.to_excel(w,sheet_name = 'magic_r_total',index = False)    
with pd.ExcelWriter('magic_PostSeason.xlsx') as w:
    magic_post_season.to_excel(w,sheet_name = 'magic_p_season',index = False)
with pd.ExcelWriter('magic_p_total.xlsx') as w:
    magic_total_P.to_excel(w,sheet_name = 'magic_p_total',index = False)
#%%
 #22-23시즌 TEAM 스텟
from nba_api.stats.endpoints import TeamYearByYearStats

teams_stats = pd.DataFrame()
for team_id in nba_teams.id:
    stats = TeamYearByYearStats(team_id).get_data_frames()[0]
    last_year = stats.loc[stats['YEAR'] =='2022-23']
    teams_stats = pd.concat([teams_stats,last_year])
#%%
with pd.ExcelWriter('2022-2023.xlsx') as w:
    teams_stats.to_excel(w,sheet_name = '2022-2023',index = False)    
    
#%%
#22-23시즌 PLAYER 스텟
last_season_stats = pd.DataFrame() 
for ID in  nba_players.id:
    player_stat = PlayerCareerStats(player_id=ID).get_data_frames()[0]
    stat = player_stat.loc[player_stat["SEASON_ID"]=="2022-23"]
    last_season_stats = pd.concat([last_season_stats,stat])
#%%
with pd.ExcelWriter('22-23_player.xlsx') as w:
    last_season_stats.to_excel(w,index = False)   

