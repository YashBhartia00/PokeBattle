# PokeBattle
pokemon battle environment and learning agent (DQN)

 **Required Dependencies :** torch, numpy, pandas
 
 **optional Dependencies :** matplotlib
 
 results on 20,000 episodes of training, best win-rate = 43%
 
### To run the train loop: 
```python
from pokebattle import GameManager, Agent

game = GameManager(poke_per_team = 3)
player = Agent(game)
player.learn()
```
### Changes made from [SAiDL Repo](https://github.com/SforAiDl/Summer-Induction-Assignment-2020/tree/master/Question%204.2/RL%20PokeBattle):
* Updated stats and moves sheets to include more variety and types of pokemon
* Included power points management for each move
* Updated damage calculation to include stats and proper bonuses (like in the game)
* fixed some bugs 