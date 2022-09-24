import numpy as np, csv, random, copy, matplotlib.pyplot as plt
from matplotlib import collections

#변수 설정
openfilename = "csv/place_jeju.csv" # 읽을 csv 파일 이름
savefilename = "results/result_jeju_statistics.png" # 저장할 그래프 그림 파일 이름 
crossrate = 0.7 # 교차율
mutaterate = 0.05 # 변이율
reverserate = 0.2 # 교차 단계에서 경로 방향을 바꾼 자손을 넘길 확률
popsize = 1000 # 한 세대의 개체 수
elitepopsize = 50 # 다음 세대로 넘길 개체 수
maxgen = 1000 # 최종 세대 수
genunit = 1 # 그림을 업데이트하는 단위 세대 수
pausetime = 0.01 # 시간 간격
endcondition = 50 # 몇 세대 동안 Highest Fitness가 유지되면 종료할 지 설정
minsize = 5 # 포함시킬 최소 관광지 수 (첫 장소는 제외함)



#csv 파일 읽어서 list로 변환
f = open(openfilename, 'r', encoding='cp949')
rdr = csv.reader(f)
citylist = []
for line in rdr:
    citylist.append(line)
f.close()
citylist[0][0] = 0
citylist = list(map(lambda x: [int(x[0]), x[1], float(x[2]), float(x[3]), float(x[4]), int(x[5]), int(x[6]), int(x[7]), int(x[8])], citylist))



#전역 변수 설정
statistics_result = []
fig = plt.figure(figsize=(8, 8))
project = plt.subplot()
gencount = 0
maxsize = len(citylist)



class City:
    def __init__(self, id):
        self.id = id
        self.name = citylist[id][1]
        self.x = citylist[id][2]
        self.y = citylist[id][3]
        self.star = citylist[id][4]
        self.review = citylist[id][5]
        self.people = citylist[id][6]
        self.price = citylist[id][7]
        self.res = citylist[id][8]

    def distance(self, city):
        dx = abs(self.x - city.x)
        dy = abs(self.y - city.y)
        distance = (dx * dx + dy * dy) ** 0.5
        return distance



class Route:
    def __init__(self, idlist=None):
        if idlist is None:
            size = random.randrange(minsize, maxsize)
            idlist = random.sample(range(1, maxsize), size)
        self.idlist = idlist
        self.size = len(idlist)
        routedistance = City(idlist[0]).distance(City(0))
        for i in range(self.size - 1):
            routedistance += City(idlist[i]).distance(City(idlist[i+1]))
        self.distance = routedistance
        self.star = sum(list(map(lambda x: City(x).star, idlist))) / self.size
        self.review = sum(list(map(lambda x: City(x).review, idlist))) / self.size
        self.people = sum(list(map(lambda x: City(x).people, idlist))) / self.size
        self.price = sum(list(map(lambda x: City(x).price, idlist)))

    def consec_res(self):
        end = 0
        for i in self.idlist:
            if City(i).res == 1:
                end += 1
            else:
                end = 0
            if end >= 3:
                return True
        return False

    def fitness(self):
        fitness = (self.star * 5000 + 
                   self.review - 
                   self.distance * 10 - 
                   self.people * 5000 - 
                   self.price * 0
                   )
        
        if self.consec_res():
            fitness == 100

        if fitness < 100:
            fitness = 100
        return fitness
    
    def crossover(self, route):
        rd = random.random()
        if rd > crossrate:
            return self, route
        size1 = self.size
        size2 = route.size
        gene1a = random.randrange(size1)
        gene1b = random.randrange(gene1a+1, size1+1)
        gene2a = random.randrange(size2)
        gene2b = random.randrange(gene2a+1, size2+1)
        seg1a = copy.deepcopy(route.idlist)
        seg2a = copy.deepcopy(self.idlist)
        for i in range(gene1a, gene1b):
            if self.idlist[i] in seg1a:
                seg1a.remove(self.idlist[i])
        for i in range(gene2a, gene2b):
            if route.idlist[i] in seg2a:
                seg2a.remove(route.idlist[i])
        seg1b = self.idlist[gene1a:gene1b]
        seg2b = route.idlist[gene2a:gene2b]
        if random.random() < reverserate:
            seg1b = seg1b[::-1]
        if random.random() < reverserate:
            seg2b = seg2b[::-1]
        leftsize1 = len(seg1a)
        leftsize2 = len(seg2a)
        cut1 = random.randrange(leftsize1+1)
        cut2 = random.randrange(leftsize2+1)
        child1 = seg1a[:cut1] + seg1b + seg1a[cut1:]
        child2 = seg2a[:cut2] + seg2b + seg2a[cut2:]
        return Route(child1), Route(child2)

    def mutate(self):
        rd = random.random()
        if rd > mutaterate:
            return self
        mutatationtype = random.randrange(3)
        if mutatationtype == 0:
            gene1 = random.randrange(self.size-1)
            gene2 = random.randrange(gene1+1, self.size)
            idlist = copy.deepcopy(self.idlist)
            id1 = idlist[gene1]
            id2 = idlist[gene2]
            idlist[gene1] = id2
            idlist[gene2] = id1
            return Route(idlist)
        elif mutatationtype == 1:
            if self.size == minsize:
                return self
            gene = random.randrange(self.size)
            idlist = copy.deepcopy(self.idlist)
            del idlist[gene]
            return Route(idlist)
        else:
            if self.size == maxsize-1:
                return self
            idlist = copy.deepcopy(self.idlist)
            nonset = set(range(1, maxsize)) - set(idlist)
            newgene = random.sample(list(nonset), 1)
            position = random.randrange(self.size+1)
            idlist = idlist[:position] + newgene + idlist[position:]
            return Route(idlist)

    def graph(self):
        idlist = self.idlist
        routelist = list(map(lambda x: citylist[x][2:4], idlist))
        linelist = []
        linelist.append([[City(0).x, City(0).y], routelist[0]])
        for i in range(self.size-1):
            linelist.append([routelist[i], routelist[i+1]])
        routegraph = collections.LineCollection(linelist, color="black")
        project.add_collection(routegraph)
        project.autoscale()

    def __repr__(self):
        idlist = self.idlist
        s = "(" + City(0).name + ", "
        for i in range(self.size-1):
            s += City(idlist[i]).name
            s += ", "
        s += City(idlist[self.size-1]).name
        s += ")"
        return s



class Population:
    def __init__(self, routelist=None):
        if routelist is None:
            routelist = []
            for i in range(popsize):
                routelist.append(Route())
        self.routelist = routelist       
        fitlist = list(map(lambda x: x.fitness(), routelist))
        self.fitlist = fitlist
        self.fitsum = sum(fitlist)       
        roulette = [0]
        for f in fitlist:
            roulette.append(roulette[-1] + f)
        roulette = roulette[1:]
        self.roulette = roulette
        sortedfitlist = sorted(fitlist, reverse = True)
        elitefitlist = sortedfitlist[:elitepopsize]
        elitepop = list(map(lambda x: routelist[fitlist.index(x)], elitefitlist))
        self.elitepop = elitepop
        self.elite = routelist[fitlist.index(max(fitlist))]
        stlist = list(map(lambda x: x.price, routelist))
        self.mean = np.mean(stlist)
        #self.std = np.std(stlist)

    def selection(self):
        rd = random.uniform(0, self.fitsum)
        for i in range(popsize):
            if self.roulette[i] > rd:
                return self.routelist[i]
        return self.routelist[-1]

    def nextgeneration(self):
        nextgen = copy.deepcopy(self.elitepop)
        for i in range((popsize - elitepopsize)//2):
            parent1 = self.selection()
            parent2 = self.selection()
            child1, child2 = parent1.crossover(parent2)
            mchild1 = child1.mutate()
            mchild2 = child2.mutate()
            nextgen.append(mchild1)
            nextgen.append(mchild2)
        return Population(nextgen)

    def plotroute(self):
        global gencount
        xlist = list(map(lambda x: x[2], citylist))
        ylist = list(map(lambda x: x[3], citylist))
        annotations = list(range(maxsize))
        window = "Project"
        title = ("Finding the best traveling route by the Genetic Algorithm\n" +
                 "Showing the route with the highest fitness")
        desc = ("Number of places: " + str(maxsize) +
                ", population size: " + str(popsize) +
                ", crossover rate: " + str(crossrate) +
                ", mutation rate: " + str(mutaterate))
        fig.canvas.manager.set_window_title(window)
        fig.suptitle(title, size=16, ha='center')
        results_text = fig.text(0, 0, "", size=10)
        plt.gca().set_aspect('equal')
        end = [1, self.elite.fitness()]
        for i in range(maxgen):
            project.clear()
            elite = self.elite
            statistics_result.append(self.mean)
            project.set_title(desc, style='italic', size=9, pad=5)
            project.scatter(xlist, ylist, color='black', alpha=0.8, edgecolor=None)
            for j, label in enumerate(annotations):
                project.annotate(label, (xlist[j], ylist[j]))
            elite.graph()
            self = self.nextgeneration()
            elite = self.elite
            gencount += 1
            if elite.fitness() == end[1]:
                end[0] += 1
            else:
                end[0] = 1
                end[1] = elite.fitness()
            results_text.set_text("Generation : " + str(i) +
                                  "\nBest Route: " + str(elite.idlist) +
                                  "\nhighest fitness: " + str(elite.fitness()) +
                                  "\naverage fitness: " + str(self.fitsum / popsize))
            if i % genunit == 0:
                plt.pause(pausetime)
            if end[0] >= endcondition:
                break
        print("Last generation:", gencount)
        print("Best route:", elite)
        print("Highest Fitness: " + str(elite.fitness()))
        print("statistics: " + str(self.mean))
        plt.show()

    def plotgraph(self):
        plt.plot(range(gencount), statistics_result, linestyle = 'solid', color = 'black')
        plt.title("statistics - generation graph")
        plt.xlabel("generation")
        plt.ylabel("statistics")
        plt.savefig(savefilename)



pop = Population()
pop.plotroute()
pop.plotgraph()