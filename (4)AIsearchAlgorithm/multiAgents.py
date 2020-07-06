from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]#The way to pick the index of best
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print(scores)
        #print(legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #print(prevFood) the return is like[FFFTF] According to pacman.py the method should return a grid and can be used for
            #if currentfood[][]==True.
        #print(successorGameState) The return is a grid . for food,G for ghost, <>^v for directions.
        #print(newPos) #it's a tuple for the coordiance.
        #print(newFood) it's a list like [FFFTF]
        #print(list(newGhostStates))
        #print(newScaredTimes) The return is the [0,1]
        #return successorGameState.getScore()
        
        # In this algorithm, I want to agent to eat the nearest food firstly,
        # And if the ghost is more the situation will be more dangerous, the score should be 
        # more influenced by the ghost part. 
        score=0
        foodScore=0
        ghostScore=0
        # The part of score for food
        foodPositions=prevFood.asList()
        minDisFood=min([manhattanDistance(newPos, foodPos) for foodPos in foodPositions])
        foodScore=2.0/(minDisFood+1)
        
        #The part of score for ghost
        ghostPositions = successorGameState.getGhostPositions()
        #ghostScore=len(ghostPositions)*1
        weights=len(ghostPositions)*1
# =============================================================================
#         for i in range(0,len(ghostPositions)):
#             if newScaredTimes[i] ==0:
#                 ghostScore=ghostScore-1/(manhattanDistance(newPos,ghostPositions[i]))
#             if newScaredTimes[i] !=0:
#                 ghostScore=ghostScore-0.5*1/(manhattanDistance(newPos,ghostPositions[i]))
#         
# =============================================================================
        ghostDis=[manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions]
        minDisGhost=min(ghostDis)
        #print(minDisGhost)
        minDisGI=[index for index in range(len(ghostDis)) if ghostDis[index]==minDisGhost]
        chosenGhostIndex=random.choice(minDisGI)
        if newScaredTimes[chosenGhostIndex] ==0:
            if ghostDis[chosenGhostIndex] >5:
                foodScore=10*foodScore
                #ghostScore=ghostScore-0.3/(ghostDis[chosenGhostIndex]+1)
            else:
                ghostScore=ghostScore-1.0/(ghostDis[chosenGhostIndex]+1)
        if newScaredTimes[chosenGhostIndex] !=0:
            ghostScore=ghostScore
        
        score=foodScore+ghostScore
        #print(score,foodScore,ghostScore)
        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
# =============================================================================
#         ###This method is for return the best value.
#         numAgent=gameState.getNumAgents()
#         def maxValue(state,depth):
#             if depth==0 or state.isWin() or state.isLose(): return self.evaluationFunction(state)
#             maxValue=-9999
#             for action in state.getLegalActions(0):
#                 successor = state.generateSuccessor(0, action)
#                 maxValue = max(maxValue, minValue(successor, depth,1))
#             return maxValue
#         def minValue(state,depth,i):
#             if depth==0 or state.isWin() or state.isLose(): return self.evaluationFunction(state)
#             value=+9999
#             for action in state.getLegalActions(i):
#                 successor = state.generateSuccessor(i, action)
#                 if i==numAgent-1:
#                     value = min(value,maxValue(successor, depth-1))
#                 else:
#                     minValue(successor,depth,i+1)
#             return value
#         
#         return maxValue(gameState,self.depth)
#         
# =============================================================================
        
        numAgent=gameState.getNumAgents()
        def maxValue(state,depth):
            if depth==0 or state.isWin() or state.isLose(): return '',self.evaluationFunction(state)
            value=[]
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value.append((action,minValue(successor, depth,1)[1]))
                maValue=max(value, key=lambda x: x[1])                
            return maValue
        def minValue(state,depth,i):
            if depth==0 or state.isWin() or state.isLose(): return '',self.evaluationFunction(state)
            value=[]
            for action in state.getLegalActions(i):
                successor = state.generateSuccessor(i, action)
                if i==numAgent-1:
                    value.append((action,maxValue(successor, depth-1)[1]))
                else:
                    value.append((action,minValue(successor,depth,i+1)[1]))
                miValue=min(value, key=lambda x: x[1])
            return miValue
        return maxValue(gameState,self.depth)[0]
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        numAgent=gameState.getNumAgents()
        def maxValue(state,depth,A,B):
            if depth==0 or state.isWin() or state.isLose(): return '',self.evaluationFunction(state)
            maValue=()
            v=-9999
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value=minValue(successor, depth,1,A,B)[1]
                v = max(v, value)
                if v==value:
                    maValue=(action,v)
                if v>B: 
                    return maValue
                A=max(A,v)
            return maValue
        def minValue(state,depth,i,A,B):
            if depth==0 or state.isWin() or state.isLose(): return '',self.evaluationFunction(state)
            miValue=()
            v=9999
            for action in state.getLegalActions(i):
                successor = state.generateSuccessor(i, action)
                if i==numAgent-1:
                    value=maxValue(successor, depth-1,A,B)[1]
                else:
                    value=minValue(successor,depth,i+1,A,B)[1]
                v=min(v,value)
                if v==value:
                    miValue=(action,v)
                if v<A: 
                    return miValue
                B=min(B,v)
            return miValue
        return maxValue(gameState,self.depth,-9999,9999)[0]
        
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        numAgent=gameState.getNumAgents()
        def maxValue(state,depth):
            if depth==0 or state.isWin() or state.isLose(): return '',self.evaluationFunction(state)
            value=[]
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                value.append((action,avgValue(successor, depth,1)))
                maValue=max(value, key=lambda x: x[1])                
            return maValue
        def avgValue(state,depth,i):
            if depth==0 or state.isWin() or state.isLose(): return self.evaluationFunction(state)
            value=[]
            for action in state.getLegalActions(i):
                successor = state.generateSuccessor(i, action)
                if i==numAgent-1:
                    value.append(maxValue(successor, depth-1)[1])
                else:
                    value.append(avgValue(successor,depth,i+1))
                meValue=sum(value)/len(value)
            return meValue
        return maxValue(gameState,self.depth)[0]
        
        
        

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I want to evaluate from 3 perspectives: 
          the distances of the 5 nearest food, 
          the distances of the ghosts,
          the distances of the capsules.
    the food part:
       if there is only one food left, take it and end the game.
    the ghost part:
        I want to ensure the safty of the agent. So the more ghosts are, 
        the more weights should be to ensure safty.
    the capsules part:
        prefer to eat a capsule to traggle the scaredTime part.
        
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    #ghostPos=ghostStates.getPosition()
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    #weights=len(ghostPositions)*1
    
    foodScore=0
    ghostScore=0
    capScore=0
    
    #for the food part.
    if len(foodPos) == 0:
        return 99999
    distances = [manhattanDistance(position, food) for food in foodPos]
    foodScore = 0.91 / (min(distances)+1)
    
    #for the ghost part
    ghostDis=[manhattanDistance(position, ghostPos) for ghostPos in ghostPositions]
    minDisGhost=min(ghostDis)
    #print(minDisGhost)
    minDisGI=[index for index in range(len(ghostDis)) if ghostDis[index]==minDisGhost]
    chosenGhostIndex=random.choice(minDisGI)
    if scaredTimes[chosenGhostIndex] ==0:
        if ghostDis[chosenGhostIndex] >6:
            foodScore=10.5*foodScore
                #ghostScore=ghostScore-0.3/(ghostDis[chosenGhostIndex]+1)
        else:
            ghostScore=-1.5/(ghostDis[chosenGhostIndex]+2)
    if scaredTimes[chosenGhostIndex] !=0:
        ghostScore=ghostScore
    
    # capsules
    capsules = currentGameState.getCapsules()
    capdistances = [manhattanDistance(position, capsule) for capsule in capsules]
    if capdistances:
        capScore = 0.98 / (min(capdistances) + 1)
    
    
    score=score+foodScore+ghostScore+capScore
    return score
    
    

# Abbreviation
better = betterEvaluationFunction

