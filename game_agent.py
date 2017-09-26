# -*- coding: utf-8 -*-
"""Custom agent for p-adv-gameplaying AIND - asaytuno
"""
import random
from math import sqrt

debug_w = print;

#Constants
#  create coordinate plane
plane =[ (x,y) for x in range(-3,4) for y in range(-3,4) ]
#  heuristic that favors the center
#  center positions have a higher utility value
#  center gets an equivalent +1 legal moves; corners get -0.167 moves
centrist_utility={(x+3,y+3):(sqrt(2*abs(32*4-x**2-y**2))-15,(x,y)) for x,y in plane}
#  returns a list tuple (utility value, point in plane, board position)

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def custom_score(game, player):
    """Calculate how centered a player is and how close to an opponent.
    Being in the center has more open moves, sticking to an opponent
    denies open moves.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    isGameEnded = end_score(game,player);
    if isGameEnded: return isGameEnded;

    centrality = centrality_score(game,player);
    proximity = proximity_score(game,player);
    return float(centrality+proximity);

def given_score(game, player):
    """The provided heuristic of open moves minus opponent moves 
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    isGameEnded = end_score(game,player);
    if isGameEnded: return isGameEnded;

    #legal moves
    moves = len(game.get_legal_moves(player));
    op_moves = len(game.get_legal_moves(game.get_opponent(player)));
    position = game.get_player_location(player);
    return float(moves-op_moves);

def end_score(game, player):
    """ +inf or -inf if game is won or lost
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    #if the game has ended
    if game.is_loser(player):
        return float("-inf")
    if game.is_winner(player):
        return float("inf")
    return float(0);

def centrality_score(game, player):
    """Utiliy for being in the center
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    isGameEnded = end_score(game,player);
    if isGameEnded: return isGameEnded;

    #assert centrist_utility[(7,7)][0] == -0.6125054300618409, "centrist map values are different"
    position = game.get_player_location(player);
    #a score on how centered the position is
    centrality = centrist_utility[position][0]; #map of values
    return float(centrality);

def proximity_score(game, player):
    """ High utility for keeping enemies close.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    isGameEnded = end_score(game,player);
    if isGameEnded: return isGameEnded;

    #points
    me = game.get_player_location(player);
    op = game.get_player_location(game.get_opponent(player));
    #a straight line distance of opponent
    x1, y1 = me;
    x2, y2 = op;
    d = sqrt((x2-x1)**2+(y2-y1)**2);
    return float(d);

def L_score(game, player):
    """ High utility for moving onto an opponents legal spots
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    isGameEnded = end_score(game,player);
    if isGameEnded: return isGameEnded;

    #points
    me = game.get_player_location(player);
    op = game.get_player_location(game.get_opponent(player));
    x1, y1 = me;
    x2, y2 = op;
    utility=0;
    #is two points in L shape
    if abs(x2-x1) is 2 and abs(y2-y1) is 1: utility=1;
    if abs(x2-x1) is 1 and abs(y2-y1) is 2: utility=1;
    return float(utility);

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        # return if there are no legal moves
        if not legal_moves: return (-1, -1);
        self.time_left = time_left
        #initialize
        utility = float("-inf");
        move = (-1,-1);
        #set minimax or alphabeta
        f = getattr(self,self.method);

        #iterative deepening
        if self.iterative is False:
            starting_depth=self.search_depth;
            ending_depth=self.search_depth;
        else:
            starting_depth=0;
            ending_depth=16;#some large number, avg game depth is 15.5
        try:
            for i in range(starting_depth,ending_depth+1):
                utility, move = f(game, i, maximizing_player=True);
        except Timeout:
            pass;
        return move;

    def minimax(self, game, depth, maximizing_player=True):
        """minimax search algorithm

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #exit search if no time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #initialize
        opponent=game.get_opponent(self)
        mypos=game.get_player_location(self);
        oppos=game.get_player_location(opponent);
        #is max? min?
        if maximizing_player is True:
            f = max;
            utility = float("-inf");
            legal_moves = game.get_legal_moves(self);
            move=mypos;
        else:
            f = min;
            utility = float("inf");
            legal_moves = game.get_legal_moves(opponent);
            move=oppos;

        #Terminal Test
        if depth <= 0 or not legal_moves:
            utility=self.score(game, self);
            return utility, move;

        #start search
        utility, move = f([ (self.minimax(game.forecast_move(m),depth-1,maximizing_player= not maximizing_player)[0], m) for m in legal_moves ]);
        return utility, move;

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """minimax search with alpha-beta pruning
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        #exit search if no time
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        #initialize
        opponent=game.get_opponent(self)
        mypos=game.get_player_location(self);
        oppos=game.get_player_location(opponent);
        #is max? min?
        if maximizing_player is True:
            f = max;
            utility = float("-inf");
            legal_moves = game.get_legal_moves(self);
            move=mypos;
        else:
            f = min;
            utility = float("inf");
            legal_moves = game.get_legal_moves(opponent);
            move=oppos;

        #Terminal Test
        if depth <= 0 or not legal_moves:
            utility=self.score(game, self);
            return utility, move;

        #start search
        ulist=[];
        for m in legal_moves:
            ulist.append((self.alphabeta(game.forecast_move(m),depth-1,alpha,beta,maximizing_player= not maximizing_player)[0],m));
            utility, move = f(ulist); #used a temp list to make max() or min() work
            if maximizing_player is True:
                if utility >= beta: #skip if larger than the minimum among siblings
                    return utility, move;
                alpha=f([alpha,utility]);
            else:
                if utility <= alpha: #skip if smaller than the maximum among siblings
                    return utility, move;
                beta=f([beta,utility]);
        return utility, move;
