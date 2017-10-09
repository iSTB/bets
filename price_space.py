import csv
import random
import pylab
import networkx as nx
import os
import math
#from neat import nn, population, statistics
import numpy

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
            #print headers
            game_n = 0
            for row in reader:
                if row[3] == '':
                    #print row
                    continue 
                game_data ={}
                for i,head in enumerate(headers):
                    if head != '':
                        try:
                            game_data[head] = row[i]
                        except:
                            print "Warning: Incomplete row " + str(row)
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
        #self.G = nx.Graph()
        self.season = season
        for team in season.teams:
            self.G.add_node(team)

        self.current_game = 0
        self.c = 0
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
        
        try:
            htg = int(game['FTHG'])
            atg = int(game['FTAG'])
        except:
            print "no htg"
            self.current_game +=1
            return

        if (at,ht) in self.G.edges():
            d = self.G.get_edge_data(at,ht)
            d['weight'] = d['weight']*0.2#+ (htg-atg)*0.01
        if (ht,at) in self.G.edges():
            d = self.G.get_edge_data(ht,at)
            d['weight'] = d['weight']*0.2#+ (htg-atg)*0.01

        if htg > atg:
            if (at,ht) in self.G.edges():
                d = self.G.get_edge_data(at,ht)
                d['weight'] =0.5 + d['weight']#+ (htg-atg)*0.01
            else:    
                self.G.add_edge(at,ht,weight=0.5)# +(htg-atg)*0.01)

        elif htg < atg:
            if (ht,at) in self.G.edges():
                d = self.G.get_edge_data(ht,at)
                d['weight'] = 0.5 +d['weight']# (atg-htg)*0.01
            else:
                self.G.add_edge(ht,at,weight=0.5)#+ (atg-htg)*0.01)
        else:
            if (ht,at) in self.G.edges():
                d = self.G.get_edge_data(ht,at)
                d['weight'] =0.5/3 + d['weight'] #+ 0.5/3 

            else:
                self.G.add_edge(ht,at,weight=(0.5/3))

            if (at,ht) in self.G.edges():
                d = self.G.get_edge_data(at,ht)
                d['weight'] = 0.5/3 + d['weight']# + 0.5/3
            else:    
                self.G.add_edge(at,ht,weight=0.5/3)
           # self.G.add_edge(at,ht)#, weight = 1./3)
           # self.G.add_edge(ht,at)#, weight =1./3)

        #for u,v,d in self.G.edges(data=True):
        #    d['weight'] = d['weight']*.999#old games less important
        self.c+=1
        self.current_game +=1
    def update_graphp(self):
        """
        takes one game from the season to update the graph
        """
        if self.current_game == len(self.season.games)-1:
            #print "All games addedd to graph"
            return -1

        game = self.season.games[self.current_game]
        ht = game['HomeTeam']
        at = game['AwayTeam']
        try:
            hto = float(game['BbMxH'])
            ato = float(game['BbMxA'])
            do = float(game['BbMxD'])
        except:
            self.current_game +=1
            return

        #ht_prob,at_prob,d_prob = to_probs([hto,ato,do])
        ht_prob,at_prob = to_probs([hto,ato])

        if (at,ht) in self.G.edges():
            d = self.G.get_edge_data(at,ht)
            d['weight'] =ht_prob #d['weight'] + 0.5 #+ (htg-atg)*0.01
        else:    
            self.G.add_edge(at,ht,weight=ht_prob)# +(htg-atg)*0.01)
            #self.G.add_edge(at,ht)#, weight = 1.)

        if (ht,at) in self.G.edges():
            d = self.G.get_edge_data(ht,at)
            d['weight'] = at_prob#d['weight'] + 0.5 #+ (atg-htg)*0.01
        else:
            self.G.add_edge(ht,at,weight=at_prob)#+ (atg-htg)*0.01)

       # for u,v,d in self.G.edges(data=True):
       #     d['weight'] = d['weight']*.999#old games less important
        self.c+=1
        self.current_game +=1

    def make_graph_to(self,l=0.5):
        to = int(round(len(self.season.games)*l))

        for _ in xrange(to):
           self.update_graph()


train_years = ['1993-94','1994-95','1995-96','1996-97','1997-98','1998-99','1999-00','2000-01','2001-02', '2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10','2010-11','2011-12','2012-13','2013-14']
leagues = ['/E0.csv','/E1.csv','/E2.csv', '/G1.csv', '/P1.csv', '/T1.csv', '/I1.csv','/N1.csv', '/F1.csv','/SC0.csv','/SP1.csv','/SP2.csv','/B1.csv','/D1.csv', '/F2.csv', '/I2.csv', '/SC1.csv','/SC2.csv','/SC3.csv', '/E3.csv']

#train_years = ['2009-10','2010-11','2011-12','2012-13','2013-14']
test_years = ['2015-16']
leagues = ['/SP1.csv']
random.shuffle(leagues)
pwin = {}
pwina = {}
pwind = {} 
rankoddsw = []
htoddsw = []

rankodds=[]
htodds = []

def to_probs(prices):
    #return [1./x for x in prices]
    total = sum([1./x for x in prices])
    return ((1./x)/total for x in prices)
def train(leagues =leagues, seasons = train_years):#['2010-11']):
    ranksp = []
    home_probs =[]
    away_probs =[]
    draw_probs = []
    for league in leagues:
        for year in seasons:
            try:
                s = season('./seasons/'+year+league)
                s.read_season()
                n = float(len(s.teams))
            except:
                print year+league + " bust"
                continue
            
            net = network(s)
            net.make_graph_to(0.333)
            while net.update_graph() != -1:
                try:
                    ranks = nx.algorithms.centrality.eigenvector_centrality(net.G)
                except:
                    print "not converged"
                    continue 
                game = s.games[net.current_game]
                ht = game['HomeTeam']
                at = game['AwayTeam']
                try:
                    hto = float(game['BbMxH'])
                    ato = float(game['BbMxA'])
                    do = float(game['BbMxD'])
                except:
                    continue
                ht_prob,at_prob,d_prob = to_probs([hto,ato,do])
               
                home_probs.append(ht_prob)
                away_probs.append(at_prob)
                draw_probs.append(d_prob)
                htrank = math.exp(ranks[ht])
                atrank = math.exp(ranks[at])
                
                ranksp.append(htrank/(atrank+htrank)) 

      
    pylab.scatter(ranksp,home_probs,color='red')
    pylab.scatter(ranksp,away_probs, color='green')
    pylab.scatter(ranksp,draw_probs, color='blue')
    fh = numpy.poly1d(numpy.polyfit(numpy.log(ranksp),home_probs,3))  
    fa = numpy.poly1d(numpy.polyfit(numpy.log(ranksp),away_probs,3))  
    fd = numpy.poly1d(numpy.polyfit(numpy.log(ranksp),draw_probs,3))

   # fh = numpy.poly1d(numpy.polyfit((ranksp),home_probs,2))  
   # fa = numpy.poly1d(numpy.polyfit((ranksp),away_probs,2))  
   # fd = numpy.poly1d(numpy.polyfit((ranksp),draw_probs,2))

    ranksp = sorted(ranksp)
    pylab.plot(ranksp,[fh(numpy.log(x)) for x in ranksp],color='black')
    pylab.plot(ranksp,[fa(numpy.log(x)) for x in ranksp],color='black')
    pylab.plot(ranksp,[fd(numpy.log(x)) for x in ranksp],color='black')

   # pylab.plot(ranksp,[fh(x) for x in ranksp],color='black')
   # pylab.plot(ranksp,[fa(x) for x in ranksp],color='black')
   # pylab.plot(ranksp,[fd(x) for x in ranksp],color='black')

    pylab.show()
    return (fh,fa,fd)

def kelly(odds,p):
    b = odds-1.
    return (b*p-(1-p))/b


def bet_game(g,s,ranks,fit,fh,fa,fd):
    game = s.games[g.current_game]
    n = len(s.teams)
    if game['HomeTeam'] == '':
        return fit
    ht = game['HomeTeam']
    at = game['AwayTeam']
    try:
        hto = float(game['BbMxH'])
        ato = float(game['BbMxA'])
        do = float(game['BbMxD'])
    except:
        return fit
    try:
        dif = math.exp(ranks[ht])/ (math.exp(ranks[ht]) + math.exp(ranks[at]))
        #dif = (ranks[ht]/)/ ((ranks[ht]/) + (ranks[at]/))
        print dif
    except:
        return fit

    try:
        htg = int(game['FTHG'])
        atg = int(game['FTAG'])
    except: 
        return fit

    
    if htg > atg:
        outcome='home'

    elif atg > htg:
        outcome='away'
    else:
        outcome='draw'

    ht_prob,at_prob,d_prob = to_probs([hto,ato,do])

    phto = fh(numpy.log(dif))
    pato = fa(numpy.log(dif))
    #pdo = 1-(phto+pato)#fd(dif)
    pdo = fd(numpy.log(dif))
    #if abs(phto - ht_prob) > 0.3:
    #    return fit
    home_bet = fit*kelly(hto,phto)*0.1
    away_bet = fit*kelly(ato,pato)*0.1
    draw_bet = fit*kelly(do,pdo)*0.1
    print home_bet,away_bet,draw_bet 
    if outcome == 'home':
        fit += home_bet*(hto-1.)
        fit -= away_bet
        fit -= draw_bet

    elif outcome=='away':
        fit += away_bet*(ato-1.)
        fit -= home_bet
        fit -= draw_bet
    else:
        fit += draw_bet*(do-1.)
        fit -= home_bet
        fit -= away_bet
    print fit
    return fit


#test_years = ['2011-12','2012-13','2013-14','2015-16']
#leagues = ['/SP1.csv']
def eval_fitness():
        fits = []
        fit = 1000.
        c = 0
        fits.append(fit)
        fh,fa,fd=train()
        for year in test_years:
            for league in leagues:
                s = season('./seasons/'+year+league)
                s.read_season()
                g = network(s)
                g.make_graph_to(0.3333)
                ranks = nx.algorithms.centrality.eigenvector_centrality(g.G)
                while True:
                    old_fit = fit
                    fit = bet_game(g,s,ranks,fit,fh,fa,fd) 
                    fits.append(fit)
                    x =  g.update_graph() 
                    if old_fit != fit:
                        c+=1
                      
                    if x == -1:
                        break
                    ranks = nx.algorithms.centrality.eigenvector_centrality(g.G)

        pylab.plot(fits)
        pylab.show()
        print 'bets: ' + str(c)
eval_fitness()
'''
for year in train_years:
    for league in leagues:
        try:
            s = season('./seasons/'+year+league)
            s.read_season()
        except:
            print year+league + " bust"
            continue
        #s = season('./seasons/'+year+league)
        s.read_season()
        print year + league
        net = network(s)
        net.make_graph_to(0.3)
        while net.update_graph() != -1:
            try:
                ranks = nx.algorithms.centrality.eigenvector_centrality(net.G)
            except:
                print "not converged"
                continue 
            game = s.games[net.current_game]
            ht = game['HomeTeam']
            at = game['AwayTeam']
            #try:
            #    hto = float(game['BbMxH'])
            #    ato = float(game['BbMxA'])
            #    do = float(game['BbMxD'])
            #except:
               # print game['BbMxH']
           #     continue
            htrank = ranks[ht]
            atrank = ranks[at]
            try:
                htg = int(game['FTHG'])
                atg = int(game['FTAG'])
            except:
                print "no score data"
                continue 
            bin = round(htrank/(htrank+atrank),2)

            if htg > atg:
                if bin not in pwin.keys():
                    pwin[bin] = (1.,1.) 
                else:
                    pwin[bin] = [x+1. for x in pwin[bin]] 
               # rankoddsw.append(atrank - htrank)
               # htoddsw.append(hto)
            else:
                if bin not in pwin.keys():
                    pwin[bin] = (0.,1.) 
                else:
                    pwin[bin] = (pwin[bin][0], pwin[bin][1]+1)

            if htg < atg:
                if bin not in pwina.keys():
                    pwina[bin] = (1.,1.) 
                else:
                    pwina[bin] = [x+1. for x in pwina[bin]] 
            else:
                if bin not in pwina.keys():
                    pwina[bin] = (0.,1.) 
                else:
                    pwina[bin] = (pwina[bin][0], pwina[bin][1]+1)

            if htg == atg:
                if bin not in pwind.keys():
                    pwind[bin] = (1.,1.) 
                else:
                    pwind[bin] = [x+1. for x in pwind[bin]] 
            else:
                if bin not in pwind.keys():
                    pwind[bin] = (0.,1.) 
                else:
                   pwind[bin] = (pwind[bin][0], pwind[bin][1]+1)
 '''
