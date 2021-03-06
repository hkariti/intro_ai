import random, util
import numpy as np
import time
from game import Agent

#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None
    self._turn_durations = []


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    """
    # measure run time
    start_time = time.time()
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    end_time = time.time()
    self._turn_durations.append(end_time - start_time)


    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current GameState (pacman.py) and the proposed action
    and returns a number, where higher numbers are better.
    """
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    return betterEvaluationFunction(successorGameState)

  def final(self, state):
      print('Average turn time: {:0.3f}'.format(np.average(self._turn_durations)))
      self._turn_durations = []

#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
  """
  return gameState.getScore()

######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
  """

  The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

  A GameState specifies the full game state, including the food, capsules, agent configurations and more.
  Following are a few of the helper methods that you can use to query a GameState object to gather information about
  the present state of Pac-Man, the ghosts and the maze:

  gameState.getLegalActions():
  gameState.getPacmanState():
  gameState.getGhostStates():
  gameState.getNumAgents():
  gameState.getScore():
  The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
  """
  pacmanPosition = gameState.getPacmanPosition()
  score = gameState.getScore()
  # Calculate ghosts score
  numAgents = gameState.getNumAgents()
  numGhosts = numAgents - 1
  capsuleDistance = min([util.manhattanDistance(capsule, pacmanPosition) for capsule in gameState.getCapsules()], default=0)
  ghostIndices = range(1, numAgents)
  getGhostDistance = lambda g: util.manhattanDistance(pacmanPosition, gameState.getGhostPosition(g))
  closestGhostIndex = min(ghostIndices, key=lambda g: getGhostDistance(g), default=None)
  if closestGhostIndex is None:
    # To avoid getting stuck in loops with no ghosts to break them
    ghostScore = util.random.randint(-10, 10)
  else:
    ghostDistance = getGhostDistance(closestGhostIndex)
    if gameState.getGhostState(closestGhostIndex).scaredTimer:
      # Scared ghosts are an opportunity
      ghostScore = 200 - ghostDistance
    else:
      # Brave ghosts are scary :(
      ghostScore = -(500 - ghostDistance)

  # Calculate food score
  food = gameState.getFood()
  foodDistance = np.inf
  for x in range(food.width):
    for y in range(food.height):
      if not food[x][y]:
        continue
      d = util.manhattanDistance(pacmanPosition, (x, y))
      if d < foodDistance:
        foodDistance = d
  foodScore = 10/foodDistance

  return score + foodScore + ghostScore

#     ********* MultiAgent Search Agents- sections c,d,e,f*********

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self._turn_durations = []

  # Generic form for getAction, used by all class children
  def _getAction(self, gameState, strategyFunc, kwargs_func=None, init_func=None, newmax_func=None):
    from game import Directions
    begin_time = time.time()
    legal_moves = gameState.getLegalActions(self.index)
    max_score = None
    best_move = Directions.STOP
    if init_func:
        init_func()
    for move in legal_moves:
      state = gameState.generateSuccessor(self.index, move)
      if kwargs_func:
          kwargs = kwargs_func()
      else:
          kwargs = {}
      score = strategyFunc(state, self.index, self.depth, **kwargs)
      if max_score is None or score > max_score:
        max_score = score
        best_move = move
        if newmax_func:
            newmax_func(score)
    end_time = time.time()
    self._turn_durations.append(end_time - begin_time)
    return best_move

  def final(self, state):
      print('Average turn time: {:0.3f}'.format(np.average(self._turn_durations)))
      self._turn_durations = []


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    return self._getAction(gameState, self._minimax)

  def _minimax(self, rootState, agentIndex, depth):
    # Handle end of game and out of depth
    legalMoves = rootState.getLegalActions(agentIndex)
    if rootState.isWin() or rootState.isLose() or not legalMoves:
      return rootState.getScore()
    if depth == 0:
      return self.evaluationFunction(rootState)

    numAgents = rootState.getNumAgents()
    # Get scores for all child states
    scores = []
    for move in legalMoves:
      nextState = rootState.generateSuccessor(agentIndex, move)
      nextAgent = (agentIndex + 1) % numAgents
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      score = self._minimax(nextState, nextAgent, nextDepth)
      scores.append(score)
    # current agent is us: max layer
    if agentIndex == self.index:
      return max(scores)
    # current agnet is not us: min layer
    return min(scores)


######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    return self._getAction(gameState, self._alphabeta, kwargs_func=self._return_alpha_beta,
            init_func=self._init_alphabeta, newmax_func=self._update_alpha)

  def _init_alphabeta(self):
      self.alpha = -np.inf

  def _update_alpha(self, score):
      self.alpha = score

  def _return_alpha_beta(self):
      return dict(alpha=self.alpha, beta=np.inf)

  def _alphabeta(self, rootState, agentIndex, depth, alpha, beta):
    # Handle end of game and out of depth
    legalMoves = rootState.getLegalActions(agentIndex)
    if rootState.isWin() or rootState.isLose() or not legalMoves:
      return rootState.getScore()
    if depth == 0:
      return self.evaluationFunction(rootState)

    numAgents = rootState.getNumAgents()
    # current agent is us: max layer
    if agentIndex == self.index:
      maxScore = -np.inf
      for move in legalMoves:
        nextState = rootState.generateSuccessor(agentIndex, move)
        nextAgent = (agentIndex + 1) % numAgents
        if nextAgent == 0:
          nextDepth = depth - 1
        else:
          nextDepth = depth
        score = self._alphabeta(nextState, nextAgent, nextDepth, alpha, beta)
        if score >= beta:
          # value from current branch will be unused, so trim it
          return np.inf
        if score > maxScore:
          maxScore = score
          if score > alpha:
            alpha = score
      return maxScore
    # current agnet is not us: min layer
    minScore = np.inf
    for move in legalMoves:
      nextState = rootState.generateSuccessor(agentIndex, move)
      nextAgent = (agentIndex + 1) % numAgents
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      score = self._alphabeta(nextState, nextAgent, nextDepth, alpha, beta)
      if score <= alpha:
        # value from current branch will be unused, so trim it
        return -np.inf
      if score < minScore:
        minScore = score
        if score < beta:
          beta = score
    return minScore

######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as choosing uniformly at random from their legal moves.
    """
    return self._getAction(gameState, self._expectimax)

  def _expectimax(self, rootState, agentIndex, depth):
    # Handle end of game and out of depth
    legalMoves = rootState.getLegalActions(agentIndex)
    if rootState.isWin() or rootState.isLose() or not legalMoves:
      return rootState.getScore()
    if depth == 0:
      return self.evaluationFunction(rootState)

    numAgents = rootState.getNumAgents()
    # Get scores for all child states
    scores = []
    for move in legalMoves:
      nextState = rootState.generateSuccessor(agentIndex, move)
      nextAgent = (agentIndex + 1) % numAgents
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      score = self._expectimax(nextState, nextAgent, nextDepth)
      scores.append(score)
    # current agent is us: max layer
    if agentIndex == self.index:
      return max(scores)
    # current agnet is not us: probabilistic layer
    # RandomGhost treats all steps with equal probabily, so calc a simple average
    return np.average(scores)

######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent
  """

  rushProb = 0.8
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
    """
    return self._getAction(gameState, self._expectimax)

  def _expectimax(self, rootState, agentIndex, depth):
    # Handle end of game and out of depth
    legalMoves = rootState.getLegalActions(agentIndex)
    if rootState.isWin() or rootState.isLose() or not legalMoves:
      return rootState.getScore()
    if depth == 0:
      return self.evaluationFunction(rootState)

    numAgents = rootState.getNumAgents()
    isPacmanLayer = agentIndex == self.index
    # Get scores for all child states
    scores = []
    movesData = {}
    for move in legalMoves:
      nextState = rootState.generateSuccessor(agentIndex, move)
      nextAgent = (agentIndex + 1) % numAgents
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      score = self._expectimax(nextState, nextAgent, nextDepth)
      movesData[move] = dict(score=score)
      # Get the distance from pacman for ghosts (needed for probability calculations)
      if not isPacmanLayer:
        pacmanPos = nextState.getPacmanPosition()
        myPos = nextState.getGhostPosition(agentIndex)
        movesData[move]["distance"] = util.manhattanDistance(myPos, pacmanPos)
    # current agent is us: max layer
    if isPacmanLayer:
      return max([ m['score'] for m in movesData.values() ])
    # current agnet is not us: probabilistic layer
    # DirectionalGhost will pick try to rush us with probability >=rushProb
    # probability calculations is similiar to the one the ghost does
    probabilities = util.Counter()
    isScared = rootState.getGhostState(agentIndex).scaredTimer > 0
    if isScared:
      chosenDist = max([ m['distance'] for m in movesData.values() ])
    else:
      chosenDist = min([ m['distance'] for m in movesData.values() ])
    numActions = len(movesData)
    numChosen = len([m for m in movesData.values() if m['distance'] == chosenDist])
    for m, d in movesData.items():
      if d['distance'] == chosenDist:
        probabilities[m] = self.rushProb/numChosen
      probabilities[m] += (1-self.rushProb)/numActions
    probabilities.normalize()
    # Calculate expected score value based on probabilities above
    expected_value = 0
    for m, p in probabilities.items():
      expected_value += p * movesData[m]['score']
    return expected_value


######################################################################################
# I: implementing competition agent

from game import Directions
class CompetitionAgent(MultiAgentSearchAgent):
  """
    Your competition agent
  """
  def __init__(self, depth=2):
    MultiAgentSearchAgent.__init__(self, depth=depth)
    self.evaluationFunction = self.betterEvaluationFunction
    self._timeLimit = 28
    self._startTime = time.time()

  def final(self, asd):
    # Restart the clock after game has finished
    self._startTime = time.time()

  def betterEvaluationFunction(self, gameState):
    pacmanPosition = gameState.getPacmanPosition()
    score = gameState.getScore()
    # Calculate ghosts score
    numAgents = gameState.getNumAgents()
    numGhosts = numAgents - 1
    capsuleDistance = min([util.manhattanDistance(capsule, pacmanPosition) for capsule in gameState.getCapsules()], default=0)
    ghostIndices = range(1, numAgents)
    getGhostDistance = lambda g: util.manhattanDistance(pacmanPosition, gameState.getGhostPosition(g))
    closestGhostIndex = min(ghostIndices, key=lambda g: getGhostDistance(g), default=None)
    if closestGhostIndex is None:
      # To avoid getting stuck in loops with no ghosts to break them
      ghostScore = util.random.randint(-10, 10)
    else:
      ghostState = gameState.getGhostState(closestGhostIndex)
      isScared = ghostState.scaredTimer > 0
      ghostDistance = getGhostDistance(closestGhostIndex)
      ghostPos = ghostState.getPosition()
      if isScared:
        # Scared ghosts are an opportunity
        ghostScore = 200/(1+ghostDistance)
      else:
        # Brave ghosts are scary :(
        ghostScore = -(500/(1+ghostDistance))

    # Calculate food score
    food = gameState.getFood()
    foodDistance = np.inf
    for x in range(food.width):
      for y in range(food.height):
        if not food[x][y]:
          continue
        d = util.manhattanDistance(pacmanPosition, (x, y))
        if d < foodDistance:
          foodDistance = d
    foodScore = 10/foodDistance

    return score + foodScore + ghostScore

  def getAction(self, gameState):
    begin_time = time.time()
    legal_moves = gameState.getLegalActions(self.index)
    max_score = -np.inf
    best_move = Directions.STOP
    for move in legal_moves:
      state = gameState.generateSuccessor(self.index, move)
      score = self._expectimax(state, self.index, self.depth)
      if score > max_score:
        max_score = score
        best_move = move
    return best_move

  def _expectimax(self, rootState, agentIndex, depth):
    # Handle end of game and out of depth
    legalMoves = rootState.getLegalActions(agentIndex)
    if rootState.isWin() or rootState.isLose() or not legalMoves:
      return rootState.getScore()
    if depth == 0 or time.time() - self._startTime > self._timeLimit:
      # Call heuristic if we're out of depth
      # Also skip fancy alg and call heuristic if we're out of time
      return self.evaluationFunction(rootState)

    numAgents = rootState.getNumAgents()
    # Get scores for all child states
    scores = []
    for move in legalMoves:
      nextState = rootState.generateSuccessor(agentIndex, move)
      nextAgent = (agentIndex + 1) % numAgents
      if nextAgent == 0:
        nextDepth = depth - 1
      else:
        nextDepth = depth
      score = self._expectimax(nextState, nextAgent, nextDepth)
      scores.append(score)
    # current agent is us: max layer
    if agentIndex == self.index:
      return max(scores)
    # current agnet is not us: probabilistic layer
    # RandomGhost treats all steps with equal probabily, so calc a simple average
    return np.average(scores)
