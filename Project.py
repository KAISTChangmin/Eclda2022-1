import numpy as np, csv, random, copy, matplotlib.pyplot as plt
from matplotlib import collections

#변수 설정
openfilename = "csv/place_jeju.csv" # 읽을 csv 파일 이름
savefilename1 = "results/result_jeju_route.png" # 저장할 경로 그림 파일 이름 
savefilename2 = "results/result_jeju_graph.png" # 저장할 그래프 그림 파일 이름
crossrate = 0.7 # 교차율
mutaterate = 0.01 # 변이율
reverserate = 0.2 # 교차 단계에서 경로 방향을 바꾼 자손을 넘길 확률
popsize = 1000 # 한 세대의 개체 수
elitepopsize = 50 # 다음 세대로 넘길 개체 수
maxgen = 1000 # 최종 세대 수
genunit = 1 # 그림을 업데이트하는 단위 세대 수
pausetime = 0.01 # 시간 간격
endcondition = 50 # 몇 세대 동안 Highest Fitness가 유지되면 종료할 지 설정
comeback = True # 처음 위치로 돌아온다면 True, 아니라면 False



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
size = len(citylist)
fitness_average_result = []
fitness_highest_result = []
fig = plt.figure(figsize=(8, 8))
project = plt.subplot()
gencount = 0



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
            idlist = random.sample(range(size), size)
        self.idlist = idlist
        routedistance = 0
        if comeback:
            routedistance += City(idlist[-1]).distance(City(idlist[0]))
        for i in range(size - 1):
            routedistance += City(idlist[i]).distance(City(idlist[i+1]))
        self.distance = routedistance
        self.star = sum(list(map(lambda x: City(x).star, idlist))) / size
        self.review = sum(list(map(lambda x: City(x).review, idlist))) / size
        self.people = sum(list(map(lambda x: City(x).people, idlist))) / size
        self.price = sum(list(map(lambda x: City(x).price, idlist)))

    def consec_res(self):
        end = 0
        for i in self.idlist:
            if City(i).res == 1:
                end += 1
            else:
                end = 0
            if end >= 4:
                return True
        return False

    def fitness(self):
        fitness = 1 / self.distance ** 2
        
        if self.consec_res():
            fitness == 0.0000000001

        if fitness < 0:
            fitness = 0
        return fitness
    
    def crossover(self, route):
        rd = random.random()
        if rd > crossrate:
            return self
        gene1 = random.randrange(size)
        gene2 = random.randrange(gene1+1, size+1)
        seg1 = copy.deepcopy(route.idlist)
        for i in range(gene1, gene2):
            seg1.remove(self.idlist[i])
        seg2 = self.idlist[gene1:gene2]
        if random.random() < reverserate:
            seg2 = seg2[::-1]
        child = seg1[:gene1] + seg2 + seg1[gene1:]
        return Route(child)

    def mutate(self):
        rd = random.random()
        if rd > mutaterate:
            return self
        gene1 = random.randrange(size-1)
        gene2 = random.randrange(gene1+1, size)
        idlist = copy.deepcopy(self.idlist)
        id1 = idlist[gene1]
        id2 = idlist[gene2]
        idlist[gene1] = id2
        idlist[gene2] = id1
        return Route(idlist)

    def graph(self):
        idlist = self.idlist
        routelist = list(map(lambda x: citylist[x][2:4], idlist))
        linelist = []
        if comeback:
            linelist.append([routelist[0], routelist[-1]])
        for i in range(size-1):
            linelist.append([routelist[i], routelist[i+1]])
        routegraph = collections.LineCollection(linelist, color="black")
        project.add_collection(routegraph)
        project.autoscale()

    def __repr__(self):
        idlist = self.idlist
        s = "("
        for i in range(size-1):
            s += City(idlist[i]).name
            s += ", "
        s += City(idlist[size-1]).name
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

    def selection(self):
        rd = random.uniform(0, self.fitsum)
        for i in range(popsize):
            if self.roulette[i] > rd:
                return self.routelist[i]
        return self.routelist[-1]

    def nextgeneration(self):
        nextgen = copy.deepcopy(self.elitepop)
        for i in range(popsize - elitepopsize):
            parent1 = self.selection()
            parent2 = self.selection()
            child = parent1.crossover(parent2)
            mchild = child.mutate()
            nextgen.append(mchild)
        return Population(nextgen)

    def plotroute(self):
        global gencount
        xlist = list(map(lambda x: x[2], citylist))
        ylist = list(map(lambda x: x[3], citylist))
        annotations = list(range(size))
        window = "Project"
        title = ("Finding the best traveling route by the Genetic Algorithm\n" +
                 "Showing the route with the highest fitness")
        desc = ("Number of places: " + str(size) +
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
            fitness_highest_result.append(elite.fitness())
            fitness_average_result.append(self.fitsum / popsize)
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
            if i % 50 == 0:
                plt.savefig("result_inter/Generation_" + str(i))
            if i % genunit == 0:
                plt.pause(pausetime)
            if end[0] >= endcondition:
                break
        print("Best route:", elite)
        print("Highest Fitness: " + str(elite.fitness()))
        plt.savefig(savefilename1)
        plt.show()

    def plotgraph(self):
        plt.plot(range(gencount), fitness_average_result, linestyle = 'solid', color = 'black', label='average')
        plt.plot(range(gencount), fitness_highest_result, linestyle = 'dashed', color = 'black', label='highest')
        plt.title("fitness - generation graph")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.legend()
        plt.savefig(savefilename2)



pop = Population()
pop.plotroute()
pop.plotgraph()