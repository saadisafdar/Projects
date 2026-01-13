# Question 2
# Simple, student-friendly implementation
# Game: Take-away game (stones)
# Rules:
# - Start with N stones
# - Players remove 1, 2, or 3 stones
# - Player who takes the last stone wins

# import math
# from collections import defaultdict

# # ---------------- GAME STATE ----------------
# class GameState:
#     def __init__(self, stones, turn):
#         self.stones = stones      # remaining stones
#         self.turn = turn          # 1 = max player, -1 = min player

#     def moves(self):
#         return [m for m in (1, 2, 3) if self.stones - m >= 0]

#     def next_state(self, move):
#         return GameState(self.stones - move, -self.turn)

#     def is_terminal(self):
#         return self.stones == 0


# # ---------------- AGENT ----------------
# class MinimaxAgent:
#     def __init__(self, adaptive=False):
#         self.adaptive = adaptive
#         self.depth = 4
#         self.prune_count = 0
#         self.opponent_moves = defaultdict(int)

#     # Evaluation function
#     def evaluate(self, state):
#         # Adaptive agent penalizes unsafe states if opponent is irrational
#         bias = -1 if self.adaptive and self.is_irrational() else 0
#         return state.turn * (10 - state.stones + bias)

#     # Detect irrational behavior using frequency
#     def is_irrational(self):
#         total = sum(self.opponent_moves.values())
#         if total < 5:
#             return False
#         most_common = max(self.opponent_moves.values())
#         return most_common / total < 0.5

#     # Minimax with Alpha-Beta pruning
#     def minimax(self, state, depth, alpha, beta):
#         if depth == 0 or state.is_terminal():
#             return self.evaluate(state)

#         if state.turn == 1:
#             best = -math.inf
#             for move in state.moves():
#                 value = self.minimax(state.next_state(move),
#                                       depth - 1, alpha, beta)
#                 best = max(best, value)
#                 alpha = max(alpha, best)
#                 if beta <= alpha:
#                     self.prune_count += 1
#                     break
#             return best
#         else:
#             best = math.inf
#             for move in state.moves():
#                 value = self.minimax(state.next_state(move),
#                                       depth - 1, alpha, beta)
#                 best = min(best, value)
#                 beta = min(beta, best)
#                 if beta <= alpha:
#                     self.prune_count += 1
#                     break
#             return best

#     # Choose best move
#     def choose_move(self, state):
#         # Adaptive depth adjustment
#         if self.adaptive and self.is_irrational():
#             self.depth = min(7, self.depth + 1)

#         best_move = None
#         best_value = -math.inf

#         for move in state.moves():
#             value = self.minimax(state.next_state(move),
#                                  self.depth, -math.inf, math.inf)
#             if value > best_value:
#                 best_value = value
#                 best_move = move

#         return best_move


# # ---------------- GAME LOOP ----------------
# def play_match(agentA, agentB, stones=15):
#     state = GameState(stones, 1)
#     agents = {1: agentA, -1: agentB}

#     while not state.is_terminal():
#         agent = agents[state.turn]
#         move = agent.choose_move(state)

#         # Track opponent behavior for Agent B
#         if state.turn == -1:
#             agentA.opponent_moves[move] += 1

#         state = state.next_state(move)

#     return state.turn   # winner


# # ---------------- EXPERIMENT ----------------
# agentA = MinimaxAgent(adaptive=False)   # Standard
# agentB = MinimaxAgent(adaptive=True)    # Adaptive

# results = {"Agent A": 0, "Agent B": 0}

# for i in range(20):
#     winner = play_match(agentA, agentB)
#     if winner == 1:
#         results["Agent A"] += 1
#     else:
#         results["Agent B"] += 1

# # ---------------- LOGS ----------------
# print("Game Results:")
# print(results)

# print("\nPruning Counts:")
# print("Agent A:", agentA.prune_count)
# print("Agent B:", agentB.prune_count)

# print("\nOpponent Move History (observed by Agent B):")
# print(dict(agentA.opponent_moves))

# print("\nAdaptive Depth Used by Agent B:", agentB.depth)










# Question 3
# Adaptive Tic Tac Toe with unstable rules

# import random

# class TicTacToe:
#     def __init__(self, size=3, win=3):
#         self.size = size
#         self.win = win
#         self.board = [['.' for _ in range(size)] for _ in range(size)]

#     def available_moves(self):
#         return [(r, c) for r in range(self.size)
#                         for c in range(self.size)
#                         if self.board[r][c] == '.']

#     def make_move(self, move, player):
#         r, c = move
#         # Allow illegal or ignored moves (human confusion)
#         if random.random() < 0.15:
#             return False
#         self.board[r][c] = player
#         return True

#     def check_win(self, player):
#         lines = []

#         for i in range(self.size):
#             lines.append(self.board[i])
#             lines.append([self.board[r][i] for r in range(self.size)])

#         lines.append([self.board[i][i] for i in range(self.size)])
#         lines.append([self.board[i][self.size - i - 1] for i in range(self.size)])

#         for line in lines:
#             if line.count(player) >= self.win:
#                 return True
#         return False


# class AdaptiveAI:
#     def __init__(self):
#         self.exploration_rate = 0.1   # uninformed search
#         self.loss_count = 0

#     def choose_move(self, game):
#         moves = game.available_moves()

#         # Uninformed search (random)
#         if random.random() < self.exploration_rate:
#             return random.choice(moves)

#         # Informed search (heuristic)
#         return self.heuristic_move(game, moves)

#     def heuristic_move(self, game, moves):
#         for move in moves:
#             r, c = move
#             game.board[r][c] = 'X'
#             if game.check_win('X'):
#                 game.board[r][c] = '.'
#                 return move
#             game.board[r][c] = '.'
#         return random.choice(moves)

#     def adapt_strategy(self):
#         # If optimal play fails, deviate more
#         self.loss_count += 1
#         if self.loss_count >= 2:
#             self.exploration_rate = min(0.5, self.exploration_rate + 0.1)


# # ---------------- SIMULATION ----------------
# game = TicTacToe(size=4, win=3)
# ai = AdaptiveAI()

# for turn in range(20):

#     # Change rules during gameplay
#     if random.random() < 0.2:
#         game.size = random.choice([3, 4])
#         game.win = random.choice([3, 4])
#         game.board = [['.' for _ in range(game.size)] for _ in range(game.size)]

#     move = ai.choose_move(game)
#     game.make_move(move, 'X')

#     if game.check_win('X'):
#         print("AI won with current rules.")
#         break
#     else:
#         ai.adapt_strategy()

# print("Final exploration rate:", ai.exploration_rate)































# Cloud Resource Allocation using CSP + Local Search
# Simple and exam-friendly implementation

# import random

# # ---------------- ALLOCATION ENGINE ----------------
# class AllocationEngine:
#     def __init__(self, services, resources):
#         self.services = services
#         self.resources = resources
#         self.assignment = {s: random.choice(resources) for s in services}
#         self.constraints = []
#         self.best_assignment = self.assignment.copy()
#         self.best_penalty = float("inf")

#     # Add a new constraint dynamically
#     def add_constraint(self, constraint):
#         self.constraints.append(constraint)

#     # Remove a constraint dynamically
#     def remove_constraint(self):
#         if self.constraints:
#             self.constraints.pop()

#     # Calculate total penalty
#     def penalty(self, assignment):
#         return sum(c(assignment) for c in self.constraints)

#     # Local search improvement
#     def improve(self):
#         for service in self.services:
#             current_resource = self.assignment[service]

#             for resource in self.resources:
#                 self.assignment[service] = resource
#                 p = self.penalty(self.assignment)

#                 # Keep best partial solution
#                 if p < self.best_penalty:
#                     self.best_penalty = p
#                     self.best_assignment = self.assignment.copy()

#                 # Accept better or equal solutions
#                 if p <= self.penalty(self.assignment):
#                     break

#             self.assignment[service] = current_resource


# # ---------------- SIMULATION ----------------
# services = ["Web", "DB", "Cache"]
# resources = ["VM1", "VM2"]

# engine = AllocationEngine(services, resources)

# # Conflicting constraints
# engine.add_constraint(lambda a: 1 if a["Web"] == a["DB"] else 0)
# engine.add_constraint(lambda a: 1 if a["Cache"] != "VM2" else 0)

# # Continuous execution
# for step in range(50):

#     # Simulate dynamic environment
#     if step == 20:
#         engine.add_constraint(lambda a: 1 if a["Web"] == "VM1" else 0)

#     if step == 35:
#         engine.remove_constraint()

#     engine.improve()

# # ---------------- OUTPUT ----------------
# print("Current Allocation:", engine.assignment)
# print("Best Allocation Found:", engine.best_assignment)
# print("Best Penalty:", engine.best_penalty)

























# Integrated AI Assistant
# Combines NLP + CSP + A* + Minimax in ONE system

# # -------- NLP MODULE --------
# class NLP:
#     def interpret(self, command):
#         if "avoid" in command:
#             return {"goal": "avoid", "target": command.split()[-1]}
#         return {"goal": "reach", "target": command.split()[-1]}

# # -------- CSP MODULE --------
# class ConstraintSystem:
#     def valid(self, plan):
#         # Constraint: agent cannot enter forbidden area
#         return "forbidden" not in plan

# # -------- A* PLANNER --------
# class Planner:
#     def plan(self, start, goal):
#         # Simplified A* path
#         return [start, "move", goal]

# # -------- ADVERSARY (MINIMAX) --------
# class Adversary:
#     def decide(self, action):
#         # Adversary blocks dangerous actions
#         if action == "trap":
#             return "wait"
#         return action

# # -------- INTEGRATED AGENT --------
# class IntegratedAgent:
#     def __init__(self):
#         self.nlp = NLP()
#         self.csp = ConstraintSystem()
#         self.planner = Planner()
#         self.enemy = Adversary()

#     def act(self, command):
#         # Step 1: NLP
#         intent = self.nlp.interpret(command)

#         # Step 2: Planning
#         plan = self.planner.plan("start", intent["target"])

#         # Step 3: Constraint checking
#         if not self.csp.valid(plan):
#             plan = ["wait"]

#         # Step 4: Adversarial decision
#         final_action = self.enemy.decide(plan[-1])

#         return final_action


# # -------- EXECUTION --------
# agent = IntegratedAgent()

# print(agent.act("reach base"))
# print(agent.act("avoid trap"))





