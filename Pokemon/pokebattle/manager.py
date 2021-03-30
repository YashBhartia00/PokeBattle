import random

import numpy as np
import pandas as pd


class GameManager:
    """
    This is the GameManager class which will act as the environment for the
    assignment.

    Some features:
        There are 2 folders: stats, moves. These contain some common pokemon
            data and moves data. You can refer to these or even add to them if
            you want to. (do note down all changes you make and mention them
            in the answer doc)
        There is the type chart. The type chart is a damage multiplier chat.
        There is a separate file to hold the Opponent class but it isn't nec
    """

    def __init__(self, stats=None, moves=None, poke_per_team=3):
        # Get the database of all pokemon (their names, types, hps, available moves and thier stats)
        self.stats = (
            pd.read_csv("data/stats.csv")
            if stats is None
            else pd.read_csv("{}".format(stats))
        )
        # Get the database of moves
        self.moves_dict = (
            pd.read_csv("data/moves.csv")
            if moves is None
            else pd.read_csv("{}".format(moves))
        )

        self.moves = self.moves_dict.copy()
        # Row corresponds to attacker, column corresponds to defender
        self.type_chart = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [2.0, 1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.5, 1.0, 2.0, 2.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                [1.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 0.5, 0.5, 2.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
                [1.0, 1.0, 0.5, 0.5, 2.0, 2.0, 0.5, 1.0, 0.5, 0.5, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
                [1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0, 1.0],
                [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 2.0, 1.0, 1.0, 0.5, 2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        )
        # Define some common lists for printing and debugging purposes
        self.types = ["NORMAL","FIGHTING","FLYING","POISON","GROUND","ROCK","BUG","GHOST","STEEL","FIRE","WATER","GRASS",
                        "ELECTRIC","PSYCHIC","ICE","DRAGON","DARK","NONE"]
        self.poke_list = list(self.stats["pokemon"])
        self.moves_list = list(self.moves["move"])

        # Replace the columns with numbers
        self.moves["move"] = range(len(self.moves))
        self.moves["type"] = pd.Series(
            [self.types.index(i) for i in self.moves["type"]]
        )
        self.moves = self.moves.to_numpy()

        # Replace string pokemon names with their respective numerical indices
        self.stats["pokemon"] = self.stats.index
        self.stats["type1"] = pd.Series( [self.types.index(i) for i in self.stats["type1"]])
        self.stats["type2"] = pd.Series( [self.types.index(i) for i in self.stats["type2"]])

        for i in range(4):
            key = "move" + str(i + 1)
            self.stats[key] = pd.Series(
                [self.moves_list.index(j) for j in self.stats[key]]
            )

        # Number of pokemon per team
        self.poke_per_team = poke_per_team

        # Initialise both teams
        self.team = self._init_team()
        self.opp_team = self._init_team()

        # Initialise the starting pokemon of each team
        self.index = random.randint(0, self.poke_per_team - 1)
        self.opp_index = random.randint(0, self.poke_per_team - 1)

        # True if it's the player's turn, False if it's the opponent's turn
        # By default, the player plays first with 50% probability
        self.turn = True if np.random.uniform() < 0.5 else False

    @property
    def action_space(self):
        """
        Defines the action space of the game. It will be the indices of all the
        values in num of moves + 2 extra actions to allow switching of pokemon
        """
        return tuple(range(6))

    def _init_team(self):
        """
        Helper function to initialise the teams
        """
        indices = random.sample(range(len(self.stats)), self.poke_per_team)
        team = np.array([self.stats.iloc[index] for index in indices]).astype(int)
        pp1 = [self.moves[team[i][9]][3] for i in range(self.poke_per_team)]
        pp2 = [self.moves[team[i][10]][3] for i in range(self.poke_per_team)]
        pp3 = [self.moves[team[i][11]][3] for i in range(self.poke_per_team)]
        pp4 = [self.moves[team[i][12]][3] for i in range(self.poke_per_team)]
        team = np.c_[team, pp1,pp2,pp3,pp4].astype(int)
        return team

    def reset(self):
        """
        Performs env.reset() like in Gym Environments
        
        Returns
        -------
        state : hstack of all stats of all 3 pokemon on player team, 
                index of current pokemons, number of faints of both team,
                and enemy pokemon's ID, types, and current HP
        
        statesize = 59
        """
        self.index = random.randint(0, self.poke_per_team - 1)
        self.opp_index = random.randint(0, self.poke_per_team - 1)
        self.turn = True if np.random.uniform() < 0.5 else False

        self.team, self.opp_team = self._init_team(), self._init_team()
        state = np.hstack((
            self.team[0], self.team[1], self.team[2],
            np.array([
                self.index,
                self.opp_index,
                self.team_faints(self.team),
                self.team_faints(self.opp_team),
                self.opp_team[self.opp_index][0],
                self.opp_team[self.opp_index][1],
                self.opp_team[self.opp_index][2], 
                self.opp_team[self.opp_index][3], 
                ]) 
            ))
        return state

    def validate_hp(self, player=True):
        """
        Validates the HP. You can add other validation checks here.

        Args:
            player (bool): True if the Player's HP needs to be checked
        """
        if player:
            hp = self.team[self.index][3]
        else:
            hp = self.opp_team[self.opp_index][3]
        return hp > 0

    def opp_step(self):
        """
        Chooses best action for the opponent AI
        """
        # The Opponent AI here basically picks the move with the highest damage
        # It won't switch until it's out of HP
        actions = self.opp_team[self.opp_index][9:13]
        valid_action = np.array([not self.opp_team[self.opp_index][13:17][i] >0 for i in range(4)])
        damages = np.array([self.moves[i][2] for i in actions])
        damages = np.ma.array(damages,mask=valid_action)
        action = np.argmax(damages)

        assert self.index in range(self.poke_per_team) and self.opp_index in range(
            self.poke_per_team
        ), "Index: {}, Opp Index: {}".format(self.index, self.opp_index)

        if action == len(self.moves):  # Switches to the pokemon to the right
            self.opp_index = (
                self.opp_index + 1 if self.opp_index < self.poke_per_team else 0
            )
            self.damage = 0
        elif action == len(self.moves) + 1:  # Switches to the pokemon to the left
            self.opp_index = (
                self.opp_index - 1 if self.opp_index > 0 else self.poke_per_team - 1
            )
            self.damage = 0
        else:
            move = self.opp_team[self.opp_index][9 + action]
            _, move_type, power, _, acc, category = self.moves[move]
            type_factor = self.type_chart[self.team[self.index][1]][ int(move_type) ] * self.type_chart[self.team[self.index][2]][ int(move_type) ]
            STAB = 1.5 if self.opp_team[self.opp_index][1] == move_type or self.opp_team[self.opp_index][2] == move_type else 1
            attack_defense = self.opp_team[self.opp_index][4] /self.team[self.index][5]
            sp_attack_defense = self.opp_team[self.opp_index][6] /self.team[self.index][7]
            
            self.damage = 0.8 *power * acc * type_factor * STAB * (attack_defense if category == "Physical" else sp_attack_defense)
            self.team[self.index][3] -= self.damage
            self.opp_team[self.opp_index][13+action]-=1
            # print("Enemy: ", self.damage, type_factor,STAB,category, attack_defense,sp_attack_defense)


    def player_step(self, action):
        """
        Step according to given action for player
        """
        if action == 4:  # Switches to the pokemon to the right
            self.index = self.index + 1 if self.index < self.poke_per_team - 1 else 0
            self.damage = 0
            type_factor = 1
        elif action == 5:  # Switches to the pokemon to the left
            self.index = self.index - 1 if self.index > 0 else self.poke_per_team - 1
            self.damage = 0
            type_factor = 1
        else:
            move = self.team[self.index][9 + action]
            _ , move_type, power, _, acc, category = self.moves[move]
            assert self.index in range(self.poke_per_team) and self.opp_index in range(self.poke_per_team)
            type_factor = self.type_chart[self.opp_team[self.opp_index][1]][int(move_type)] * self.type_chart[self.opp_team[self.opp_index][2]][int(move_type)]
            STAB = 1.5 if self.team[self.index][1] == move_type or self.team[self.index][2] == move_type else 1
            attack_defense = self.team[self.index][4] /self.opp_team[self.opp_index][5]
            sp_attack_defense = self.team[self.index][6] /self.opp_team[self.opp_index][7]
            
            self.damage = 0.8 *power * acc * type_factor * STAB * (attack_defense if category == "Physical" else sp_attack_defense)
            self.opp_team[self.opp_index][3] -= self.damage
            self.team[self.index][13+action]-=1


    def team_faints(self, team):
        """
        team: which team to evaluate

        returns number of fainted pokemon 
        
        """
        fainted = 0
        for pokemon in team:
            if pokemon[3]>0:
                fainted+=1
        return fainted

    
    def step(self, action):
        """
        Performs env.step() like in Gym Environments for the Agent

        Args:
            action (np.ndarray): Action to be taken
                0, 1, 2, 3: Moves of the pokemon
                4: Switch to the pokemon on the left (if there's no pokemon on
                    the left, it'll switch to the pokemon on the extreme right)
                5: Switch to the pokemon on the right (opposite of the above
                    in the extreme case)
        """
        opp_faint, faint = False, False
        reward = 0
        
        #step and update environment
        if not self.turn:
            self.opp_step()
            if not self.validate_hp():
                self.index = self.index + 1 if self.index < self.poke_per_team - 1 else 0
                faint = True

            self.player_step(action)
            if not self.validate_hp(False):
                self.opp_index =  self.opp_index + 1 if self.opp_index < self.poke_per_team - 1 else 0
                opp_faint = True

        else:
            self.player_step(action)
            if not self.validate_hp(False):
                self.opp_index =  self.opp_index + 1 if self.opp_index < self.poke_per_team - 1 else 0
                opp_faint = True

            self.opp_step()
            if not self.validate_hp():
                self.index = self.index + 1 if self.index < self.poke_per_team - 1 else 0
                faint = True

        self.turn = not self.turn

        # rewards = hp % + team_size*faint (if faint this turn)
        for indexs in range(self.poke_per_team):
            reward += (
                    self.team[indexs][3]/self.stats["hp"][self.team[indexs][0]] 
                    - self.opp_team[indexs][3]/self.stats["hp"][self.opp_team[indexs][0]]
                    )
        reward += self.poke_per_team *(1 if opp_faint else (-1 if faint else 0))
        #normalize Reward
        reward = reward/(3*self.poke_per_team)
        faint, opp_faint = False, False

        next_state = np.hstack((
            self.team[0], self.team[1], self.team[2],
            np.array([
                self.index,
                self.opp_index,
                self.team_faints(self.team),
                self.team_faints(self.opp_team),
                self.opp_team[self.opp_index][0],
                self.opp_team[self.opp_index][1],
                self.opp_team[self.opp_index][2], 
                self.opp_team[self.opp_index][3], 
                ]) 
            ))

        # info contains who won (true if player wins)
        # done contains if game is completed
        info = False
        done = False
        if not self.validate_hp(False):
            reward += 1
            done = True
            info = True  
            return next_state, reward, done, info
        if not self.validate_hp():
            reward -= 1
            done = True
            info = False 

        return next_state, reward, done, info
