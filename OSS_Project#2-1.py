import pandas as pd

# Print top 10 players each year received as a argument
def top10In(year, dataset_df):

    condition = dataset_df['year'] == year

    print('in', year)
    top_10_type = ['H', 'avg', 'HR', 'OBP']

    year_df = pd.DataFrame()
    for type in top_10_type:
        df = dataset_df.loc[condition].sort_values(by=[type], ascending = False).head(10)[['batter_name', type]]
        df = df.reset_index(drop=True).rename(columns={'batter_name':'name'})
        year_df = pd.concat([year_df, df], axis=1)

    print(year_df)
# Run top10In function by receiving dataset
def topPlayers(dataset_df):

    for year in range(2015, 2019):
        top10In(year, dataset_df)


# Print the highest war in each position by receiving dataset
def theHighestWar(dataset_df, year):

    print()

    positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    for position in positions:
        condition = (dataset_df['year'] == year) & (dataset_df['cp'] == position)
        globals()['the_highest_war_in_{}'.format(position)] = dataset_df.loc[condition].sort_values(by=['war'], ascending=False).head(1)['batter_name']
        globals()['the_highest_war_in_{}'.format(position)].index = [position]
        globals()['the_highest_war_in_{}'.format(position)].index.name = 'position'

    result = pd.Series([])

    for top_player in globals():
        if(top_player.find('the_highest_war_in_')) != -1:
            result = pd.concat([result, globals()[top_player]])
        else:
            pass
    
    result.name = "in 2018, the highest war in each position"
    print(result)

# Calculating the highest correlation with salary by receiving dataset
def Correlation(dataset_df):

    print()
    
    correlation_set = dataset_df[['R','H','HR','RBI','SB','war','avg','OBP','SLG']]
    print("The highest correlation with salary is :", correlation_set.corrwith(dataset_df['salary']).sort_values(ascending = False).head(1).index[0])

if __name__=='__main__':
    data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
    topPlayers(data_df)         # 1
    theHighestWar(data_df, 2018)# 2
    Correlation(data_df)        # 3