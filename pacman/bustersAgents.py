from __future__ import print_function
from argparse import _CountAction
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
from cmath import exp
from math import dist
from multiprocessing.spawn import import_main_path
from operator import indexOf
import util
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

from wekaI import Weka

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs

class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

        self.distancer = Distancer(gameState.data.layout, False)

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

    def printLineHeader(self, fileName):
        columnHeader = f"@relation {fileName} \n" \
                        "\n" \
                        "@attribute pmPositionX numeric \n" \
                        "@attribute pmPositionY numeric \n" \
                        "@attribute legalNorth {True,False} \n" \
                        "@attribute legalSouth {True,False} \n" \
                        "@attribute legalWest {True,False} \n" \
                        "@attribute legalEast {True,False} \n" \
                        "@attribute g0PositionX numeric \n" \
                        "@attribute g0PositionY numeric \n" \
                        "@attribute g0Distance numeric \n" \
                        "@attribute g0Direction {Stop,North,South,West,East} \n" \
                        "@attribute g1PositionX numeric \n" \
                        "@attribute g1PositionY numeric \n" \
                        "@attribute g1Distance numeric \n" \
                        "@attribute g1Direction {Stop,North,South,West,East} \n" \
                        "@attribute g2PositionX numeric \n" \
                        "@attribute g2PositionY numeric \n" \
                        "@attribute g2Distance numeric \n" \
                        "@attribute g2Direction {Stop,North,South,West,East} \n" \
                        "@attribute g3PositionX numeric \n" \
                        "@attribute g3PositionY numeric \n" \
                        "@attribute g3Distance numeric \n" \
                        "@attribute g3Direction {Stop,North,South,West,East} \n" \
                        "@attribute scoreCurrent numeric \n" \
                        "@attribute scoreFuture numeric \n" \
                        "@attribute pmDirection {North,South,West,East} \n" \
                        "\n" \
                        "@data\n"

        return(columnHeader)

    def printLineData(self, gameState):
        if (not isinstance(self, WekaAgent) and gameState.data.agentStates[0].getDirection() == Directions.STOP):
            return("")

        lineData = f"{gameState.getPacmanPosition()[util.X]},{gameState.getPacmanPosition()[util.Y]}"

        for i in Directions.ALL:
            lineData += f",{i in gameState.getLegalPacmanActions()}"

        livingGhosts = gameState.getLivingGhosts()[1:]
        ghostsData = ""

        MAX_NUM_GHOSTS = 4
        for indexLivingGhost in range(MAX_NUM_GHOSTS):
            if (indexLivingGhost >= len(livingGhosts)) or (not livingGhosts[indexLivingGhost]):
                ghostsData += f"1000,1000,1000,{Directions.STOP},"
            else:
                ghostsData += f"{gameState.getGhostPositions()[indexLivingGhost][util.X]},{gameState.getGhostPositions()[indexLivingGhost][util.Y]},"
                ghostsData += f"{self.distancer.getDistance(gameState.getPacmanPosition(), gameState.getGhostPositions()[indexLivingGhost])},"
                ghostsData += f"{Directions.STOP}," if gameState.getGhostDirections().get(indexLivingGhost) is None else f"{gameState.getGhostDirections().get(indexLivingGhost)},"
                
        lineData += f",{ghostsData[:-1]}"

        lineData += f",{gameState.getScore()}"

        try:
            lineData += f",{gameState.generatePacmanSuccessor(gameState.data.agentStates[0].getDirection()).getScore()}"  
        except Exception:
            lineData += f",{gameState.getScore() - 1}"
        
        lineData += f",{gameState.data.agentStates[0].getDirection()}\n"

        return(lineData)

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer, manhattanDistance
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        
    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    ''' Print the layout'''  
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
        
    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move
        
class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.countActions = 0
        
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food
    
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table
    
    def chooseAction(self, gameState):
        self.countActions += 1
        
        distance = self.getClosestGhostDistance(gameState)

        move = Directions.STOP

        if distance[util.Y] != 0:
            move = Directions.SOUTH if distance[util.Y] > 0 else Directions.NORTH
        elif distance[util.X] != 0:
            move = Directions.WEST if distance[util.X] > 0 else Directions.EAST

        if move not in gameState.getLegalPacmanActions():
            legalActions = gameState.getLegalPacmanActions()[:-1]
            move = legalActions[random.randint(0, len(legalActions) - 1)]

        return move

    def getClosestGhostDistance(self, gameState):
        ghostDistances = gameState.data.ghostDistances

        for index in [i for i, dist in enumerate(ghostDistances) if dist == None]: 
            ghostDistances[index] = 1000000

        indexClosestGhost = [i for i, dist in enumerate(ghostDistances) if dist == min(ghostDistances)][0]

        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPositions()

        xDist = pacmanPosition[util.X] - ghostPosition[indexClosestGhost][util.X]
        yDist = pacmanPosition[util.Y] - ghostPosition[indexClosestGhost][util.Y]

        return([xDist, yDist])

class ProAgent(BasicAgentAA):
    def getDistanceAvoidingWallsFromPacmanToGhost(self, gameState, index):
        return self.distancer.getDistance(gameState.getPacmanPosition(), gameState.getGhostPositions()[index])

    def getClosestGhostIndex(self, gameState):
        livingGhosts = gameState.getLivingGhosts()[1:]
        distancesGhosts = [1000000] * len(livingGhosts)

        for i in range(len(livingGhosts)):
            if (livingGhosts[i]):
                distancesGhosts[i] = self.getDistanceAvoidingWallsFromPacmanToGhost(gameState, i)

        return [i for i, dist in enumerate(distancesGhosts) if dist == min(distancesGhosts)][0]

    def chooseAction(self, gameState):
        self.countActions += 1
        
        indexClosestGhost = self.getClosestGhostIndex(gameState)

        distanceAfterMove = [1000000] * len(Directions.ALL)
        for index in range(len(Directions.ALL)):
            if Directions.ALL[index] not in gameState.getLegalPacmanActions():
                continue

            nextGameState = gameState.generateSuccessor(0, Directions.ALL[index])
            
            distanceAfterMove[index] = self.getDistanceAvoidingWallsFromPacmanToGhost(nextGameState, indexClosestGhost)
            

        indexBestMove = [i for i, dist in enumerate(distanceAfterMove) if dist == min(distanceAfterMove)][0]

        return Directions.ALL[indexBestMove]

class WekaAgent(BustersAgent):
    def __init__( self, model , trainingData, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        super().__init__(index, inference, ghostAgents, observeEnable, elapseTimeEnable)

        self.weka = Weka()
        self.weka.start_jvm()   

        self.model = model
        self.trainingData = trainingData  

    def __del__(self):
        self.weka.stop_jvm()

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.countActions = 0
    
    def chooseAction(self, gameState):
        data = self.printLineData(gameState).split(",")[:-1]

        colToDelete = [-1, 21, 17, 13, 9, 5, 4, 3, 2]
        for i in colToDelete:
            del data[i]

        for i in range(len(data)):
            try:
                data[i] = int(data[i])
            except:
                pass

        move = self.weka.predict(self.model, data, self.trainingData)

        return move if move in gameState.getLegalPacmanActions() else random.choice(gameState.getLegalPacmanActions())
