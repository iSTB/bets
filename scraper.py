import csv




class season(object):
    """
    Makes dict - games = {0:{HomeTeam:Wolves,AwayTeam:Wigan,...}, {1:{}}
    """
    def __init__(self,path):
            self.path = path
            #self.season = season
            #self.league = league

            self.games = {}
            self.teams = []
            self.header = []

    def read_season(self):
        with open(self.path,mode='r') as data:
            reader = csv.reader(data)
            headers= next(reader)
            game_n = 0
            for row in reader:
                game_data ={}
                for i,head in enumerate(headers):
                    game_data[head] = row[i]
                self.games[game_n] = game_data
                game_n+=1
            for gamen in self.games.keys():
                self.teams.append(self.games[gamen]['HomeTeam'])

            self.teams = list(set(self.teams))
r = season('./seasons/2011-12/E0.csv')
r.read_season()

