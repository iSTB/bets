import csv
import networkx as nx
import os
from neat import nn, population, statistics


class season(object):
    """
    Makes dict - games = {0:{HomeTeam:Wolves,AwayTeam:Wigan,...}, {1:{}}
    """
    def __init__(self,path):
            self.path = path

            self.games = {}
            self.teams = []

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



class network(object):
    """
    takes a season, makes a netowrk
    """
    def __init__(self,season):
        self.G = nx.DiGraph()
        self.season = season
        for team in season.teams:
            self.G.add_node(team)

        self.current_game = 0
    def update_graph(self):
        """
        takes one game from the season to update the graph
        """
        if self.current_game == len(self.season.games)-1:
            #print "All games addedd to graph"
            return -1

        game = self.season.games[self.current_game]
        ht = game['HomeTeam']
        at = game['AwayTeam']
        
        htg = int(game['FTHG'])
        atg = int(game['FTAG'])

        if htg > atg:
            self.G.add_edge(at,ht,weight=1.)

        elif htg < atg:
            self.G.add_edge(ht,at,weight=1.)

        
        else:
            self.G.add_edge(ht,at,weight=1./3)
            self.G.add_edge(at,ht,weight=1./3)

        for u,v,d in self.G.edges(data=True):
            d['weight'] = d['weight']*0.95 #old games less important

        self.current_game +=1

    def make_graph_to(self,l=0.5):
        to = int(round(len(self.season.games)*l))

        for _ in xrange(to):
           self.update_graph()

train_years = ['2008-09','2009-10','2010-11','2011-12','2012-13','2013-14','2014-15','2015-16']

def bet_game(net,g,s,ranks,fit):
    game = s.games[g.current_game]
    if game['HomeTeam'] == '':
        return fit
    ht = game['HomeTeam']
    at = game['AwayTeam']
    #print game['LBH']
    try:
        hto = float(game['BbMxH'])
        ato = float(game['BbMxA'])
        do = float(game['BbMxD'])
    except:
        return fit
        hto = 1.0 
        ato = 1.0
        do = 1.0
        print game
        
    odds = [hto,ato,do]
    htrank = ranks[ht]*10
    atrank = ranks[at]*10
    inputs = [htrank,atrank,hto,ato,do]
    output = net.serial_activate(inputs)[0]
   # bet = output.index(max(output)) #0 = home_win, 1 = away_win, 2 = draw, 3 = dont bet

    if output >=0 and output < 0.25:
        bet = 0
    elif output >=0.25 and output < 0.5:
        bet = 1
    elif output >=0.5 and output< 0.75:
        bet = 2
    else:
        bet = 3
    
    #print output
    try:
        htg = int(game['FTHG'])
        atg = int(game['FTAG'])
    except: 
        return fit
   
    if htg > atg:
        outcome = 0

    elif htg < atg:
        outcome = 1
    else:
        outcome = 2

    if outcome == bet:
        fit += 1*(odds[outcome]-1.)
    elif bet ==3:
        fit = fit
    else:
      #  if fit < 100:
        fit -= 1.0
       # else:
        #    fit -= (fit*0.02)
  #  print (htg,atg),bet,fit,odds[outcome]
    return fit


leagues=['/E0.csv','/SP1.csv']

def eval_fitness(genomes):
    for ge in genomes:
        net = nn.create_feed_forward_phenotype(ge)
        fit = 100

        for league in leagues:
            for year in train_years:
                s = season('./seasons/'+year+league)
                s.read_season()
                g = network(s)
                g.make_graph_to()
                ranks = nx.pagerank(g.G)
                while True:
                    fit = bet_game(net,g,s,ranks,fit) 
                    x =  g.update_graph() 
                      
                    if x == -1:
                        break
                    
                    ranks = nx.pagerank(g.G)
            ge.fitness = fit  
        print ge.fitness



pop = population.Population('conf')

pop.run(eval_fitness, 100)
statistics.save_stats(pop.statistics)
statistics.save_species_count(pop.statistics)
statistics.save_species_fitness(pop.statistics)


to_test = []
winner = pop.statistics.best_genome()
wn = nn.create_feed_forward_phenotype(winner)


test_years = ['2008-09','2009-10','2010-11','2011-12','2012-13','2013-14','2015-16']

def test_winner(w,league='/E0.csv',test_years=test_years): 
        fit = 100
        for year in test_years:
            s = season('./seasons/'+year+league)
            s.read_season()
            g = network(s)
            g.make_graph_to()
            ranks = nx.pagerank(g.G)
            while True:
                fit = bet_game(w,g,s,ranks,fit) 
                print fit
                x =  g.update_graph() 
                  
                if x == -1:
                    break
                
                ranks = nx.pagerank(g.G)
        return fit


print test_winner(wn)
                    
#r = season('./seasons/2011-12/E0.csv')
#r.read_season()
#g = network(r)
#g.make_graph_to()
#print g.G


