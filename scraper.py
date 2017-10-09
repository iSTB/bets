import csv
import random
import pylab
import networkx as nx
import os
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
    def __init__(self,season,space = 'Result'):
        self.space = space
        self.G = nx.DiGraph()
        self.season = season
        for team in season.teams: #initing nodes of network
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
            return
        if htg > atg:
            if (at,ht) in self.G.edges():
                d = self.G.get_edge_data(at,ht)
                d['weight'] =0.5 #d['weight'] + 0.5 #+ (htg-atg)*0.01
            else:    
                self.G.add_edge(at,ht,weight=0.5)# +(htg-atg)*0.01)
            #self.G.add_edge(at,ht)#, weight = 1.)

        elif htg < atg:
            if (ht,at) in self.G.edges():
                d = self.G.get_edge_data(ht,at)
                d['weight'] = 0.5#d['weight'] + 0.5 #+ (atg-htg)*0.01
            else:
                self.G.add_edge(ht,at,weight=0.5)#+ (atg-htg)*0.01)
            #self.G.add_edge(ht,at)# weight = 1.)

        
        else:
            if (ht,at) in self.G.edges():
                d = self.G.get_edge_data(ht,at)
                d['weight'] =0.5/3#d['weight'] + 0.5/3 

            else:
                self.G.add_edge(ht,at,weight=(0.5/3))

            if (at,ht) in self.G.edges():
                d = self.G.get_edge_data(at,ht)
                d['weight'] = 0.5/3#d['weight'] + 0.5/3
            else:    
                self.G.add_edge(at,ht,weight=0.5/3)
           # self.G.add_edge(at,ht)#, weight = 1./3)
           # self.G.add_edge(ht,at)#, weight =1./3)
        for u,v,d in self.G.edges(data=True):
            d['weight'] = d['weight']*.999#old games less important
        self.c+=1
        self.current_game +=1

    def make_graph_to(self,l=0.5):
        to = int(round(len(self.season.games)*l))
        for _ in xrange(to):
           self.update_graph()
#train_years = ['2008-09','2009-10','2010-11','2011-12']

#train_years = ['1993-93','1994-95','1995-96','1996-97','1997-98','1998-99','1999-00','2000-01','2001-02', '2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10','2010-11','2011-12','2012-13','2013-14']
train_years = ['1993-94','1994-95','1995-96','1996-97','1997-98','1998-99','1999-00','2000-01','2001-02', '2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10','2010-11']#'2011-12','2012-13','2013-14']
#train_years = ['2000-01','2001-02', '2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10','2010-11']#'2011-12','2012-13','2013-14']
#train_years = ['1993-94','1994-95','1995-96','1996-97','1997-98','1998-99','1999-00','2000-01','2001-02', '2002-03','2003-04','2004-05','2005-06','2006-07','2007-08','2008-09','2009-10']
leagues = ['/E0.csv','/E1.csv','/E2.csv', '/G1.csv', '/P1.csv', '/T1.csv', '/I1.csv','/N1.csv', '/F1.csv','/SC0.csv','/SP1.csv','/SP2.csv','/B1.csv','/D1.csv', '/F2.csv', '/I2.csv', '/SC1.csv','/SC2.csv','/SC3.csv', '/E3.csv']
#leagues = ['/D1.csv','/E0.csv','/E1.csv']


random.shuffle(leagues)
pwin = {}
pwina = {}
pwind = {} 
rankoddsw = []
htoddsw = []

rankodds=[]
htodds = []

for year in train_years:
    for league in leagues:

       # s = season('./seasons/'+year+league)
       # s.read_season()
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
 
Xh = []
Yh = []

Xa = []
Ya = []

Xd = []
Yd = []

for key in sorted(pwin.keys()):
    if key >= 0.85 or key <= 0.15:
        continue
    Xh.append(key)
    Yh.append(pwin[key][0]/pwin[key][1])   
    Xa.append(key)
    Ya.append(pwina[key][0]/pwina[key][1])   
    Xd.append(key)
    Yd.append(pwind[key][0]/pwind[key][1])   
    pwin[key] = pwin[key][0]/pwin[key][1]   
    pwina[key] = pwina[key][0]/pwina[key][1]  
    pwind[key] = pwind[key][0]/pwind[key][1]  
fh = numpy.poly1d(numpy.polyfit(Xh,Yh,2))  
fa = numpy.poly1d(numpy.polyfit(Xa,Ya,2))  
fd = numpy.poly1d(numpy.polyfit(Xd,Yd,2))  

pylab.plot(Xh,Yh,'bo', label='Home Team')
pylab.plot(Xh,[fh(x) for x in Xh])

pylab.plot(Xa,Ya,'go', label='Away Team')
pylab.plot(Xa,[fa(x) for x in Xa])

pylab.plot(Xd,Yd,'ro', label='Draw')
pylab.plot(Xd,[fd(x) for x in Xd])
pylab.legend()
pylab.xlabel('rank(ht)/(rank(ht)+rank(at))')
pylab.ylabel('Probability of win')
pylab.show()

#pylab.scatter(rankodds,htodds,color='red')
#pylab.scatter(rankoddsw,htoddsw, color='green')
#pylab.show()
  
draw_confs = {}
home_confs = {}
away_confs = {}
def bet_game(g,s,ranks,fit):
    game = s.games[g.current_game]
    if game['HomeTeam'] == '':
        return fit
    ht = game['HomeTeam']
    at = game['AwayTeam']
    try:
        hto = float(game['BbMxH'])
        ato = float(game['BbMxA'])
        do = float(game['BbMxD'])
    except:
        print "couldnt bet"
        return fit
    try:
        dif = ranks[ht]/ (ranks[ht] / ranks[at] +0.000000000001)
    except:
        return fit
    #dif = htr/(atr+htr+0.00000000000001)

    bin = round(dif,2)
  #  if bin >= 1.6 or bin <= 0.5:
  #      return fit

    phto = fh(dif)
    pato = fa(dif)
    pdo = fd(dif)

    #print phto
    tot = phto + pato + pdo
    phto = phto/tot
    pato = pato/tot
    pdo = pdo/tot

    try:
        phto = pwin[bin]
        pato = pwina[bin]
        pdo = 1-(phto+pato)
        #tot = phto + pato + pdo
   #     phto = phto/tot
   #     pato = pato/tot
   #     pdo = pdo/tot
    except:
 #       phto = phto/tot
 #       pato = pato/tot
 #       pdo = pdo/tot
        return fit
        #phto = fh(dif)
        #pato = fa(dif)
        #pdo = fd(dif)
    conf = 0.25
    
    
    perc = 0.025
    
      
    bet= 'n'

    bets = [('h', phto - 1./hto),('a', pato - 1./ato)]#,('d', pdo - 1./do)]
    maxi = max(bets,key=lambda item:item[1])


    
    if maxi[1] >= 0.01 and maxi[1] <= 0.22:
        bet = maxi[0]
    print bet


    #print hto 
    #if (1./hto + conf) <phto:
    #    bet = 'h'

       # print 1./hto, phto
    #if (1./ato +conf)<pato:
       # bet = 'a'
       # print 1./ato, pato

    #if (1./do +conf)<pdo:
       # bet = 'd'
    #print bet 
#    if 1./hto + conf <phto:
 #       bet = 'hl'
 #       print 1./hto, phto

  #  if 1./ato +conf <pato:
   #     bet = 'al'
    #    print 1./ato, pato
    
         
    try:
        htg = int(game['FTHG'])
        atg = int(game['FTAG'])
    except: 
        return fit
  
   # hp = hto/(1./hto+1./ato+1/do)
   # ap = ato/(1./hto+1/ato+1./do)
   # dp = do/(1./hto+1./ato+1./do)
    hconf = round((phto - 1./hto),1)
    aconf = round((pato - 1./ato),1)
    dconf = round((pdo - 1./do),1)

    #hconf = round(phto-hp,1)
    #aconf = round(pato-ap,1)
    #dconf = round(pdo-dp,1)
    if htg > atg:
        outcome = 'h'
        if hconf not in home_confs.keys():
            home_confs[hconf] = 1.* (hto-1)
        else:
            home_confs[hconf] += 1.* (hto-1)


        if aconf not in away_confs.keys():
            away_confs[aconf] = -1.
        else:
            away_confs[aconf] -= 1.

        if dconf not in draw_confs.keys():
            draw_confs[dconf] = -1.
        else:
            draw_confs[dconf] -= 1.

    elif htg < atg:
        outcome = 'a'
        if hconf not in home_confs.keys():
            home_confs[hconf] = -1.
        else:
            home_confs[hconf] -= 1.



        if aconf not in away_confs.keys():
            away_confs[aconf] = 1.* (ato-1)
        else:
            away_confs[aconf] += 1.* (ato-1)



        if dconf not in draw_confs.keys():
            draw_confs[dconf] = -1.
        else:
            draw_confs[dconf] -= 1.

    else:
        outcome = 'd'
        if hconf not in home_confs.keys():
            home_confs[hconf] =-1
        else:
            home_confs[hconf] -=1


        if aconf not in away_confs.keys():
            away_confs[aconf] =-1
        else:
            away_confs[aconf] -=1
    

        if dconf not in draw_confs.keys():
            draw_confs[dconf] = 1.*(do-1)
        else:
            draw_confs[dconf] += 1.*(do-1)

    if outcome == bet:
        if bet =='h':
            fit += 1. * (hto-1)
            #fit += (fit*perc)* (hto-1)
           # if conf not in home_confs.keys():
           #     home_confs[conf] = 1.* (hto-1)
           # else:
           #     home_confs[conf] += 1.* (hto-1)
           # fit += 1.*(hto-1)       
        elif bet=='a':
           # if conf not in away_confs.keys():
           #     away_confs[conf] = 1.* (ato-1)
           # else:
            #    away_confs[conf] += 1.* (ato-1)
            fit += 1.*(ato-1)
        else:
            #fit += (fit*perc)*(do-1)
            #if conf not in draw_confs.keys():
            #    draw_confs[conf] = 1.* (do-1)
            #else:
            #    draw_confs[conf] += 1.* (do-1)
            fit += 1.*(do-1)

    elif bet =='n':
        fit = fit
    else:      
       # if bet == 'h':
       #     fit -= (fit*perc)
       # elif bet == 'a':
       #     fit -= (fit*perc)
       # else:
       #     fit -= (fit*perc)
       fit -= 1.
    print fit
    return fit


test_years = ['2011-12','2012-13','2013-14','2015-16']
#test_years = ['2015-16']
def eval_fitness():
        fit = 1000
        c = 0
        for year in test_years:
            for league in leagues:
                s = season('./seasons/'+year+league)
                s.read_season()
                g = network(s)
                g.make_graph_to(0.3)
                ranks = nx.algorithms.centrality.eigenvector_centrality(g.G)
                while True:
                    #if fit != bet_game(g,s,ranks,fit):
                    #    c+=1 
                    old_fit = fit
                    fit = bet_game(g,s,ranks,fit) 
                    x =  g.update_graph() 
                    if old_fit != fit:
                        c+=1
                      
                    if x == -1:
                        break
                    ranks = nx.algorithms.centrality.eigenvector_centrality(g.G)
        print fit
        print 'bets: ' + str(c)

eval_fitness()

xh = []
yh=[]

xa = []
ya =[]
xd =[]
yd = []
made_p = 0
made_m=0
for conf in sorted(home_confs.keys()):
    xh.append(conf)
    yh.append(home_confs[conf])
    if conf > 0:
        made_p += home_confs[conf]
    else:
        made_m += home_confs[conf]
for conf in sorted(away_confs.keys()):
    xa.append(conf)
    ya.append(away_confs[conf])
    if conf > 0:
        made_p += away_confs[conf]
    else:
        made_m += away_confs[conf]

for conf in sorted(draw_confs.keys()):
    xd.append(conf)
    yd.append(draw_confs[conf])
    if conf > 0:
        made_p += draw_confs[conf]
    else:
        made_m += draw_confs[conf]

pylab.plot(xh,yh,label='Home')
pylab.plot(xa,ya,label='Away')
pylab.plot(xd,yd,label='Draw')
pylab.legend()
pylab.xlabel('Expected RoI')
pylab.ylabel('Actual Return')
pylab.show()
print "made over 0 conf:" + str(made_p)
print "made less 0 conf:" + str(made_m)
'''
pop = population.Population('conf')

pop.run(eval_fitness, 10)
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

'''
