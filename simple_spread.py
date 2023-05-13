import numpy as np

from env.env_utils.core import Agent, Landmark,Obstacle, World
from env.env_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_obstacle = 0
        world.collaborative = False
        # add agents
        world.agent = Agent()
        world.agent.name = f'agent'
        world.agent.collide = True
        world.agent.size = 0.05
        # add landmarks
        world.landmark = Landmark()
        world.landmark.name = 'landmark'
        world.landmark.collide = False
        world.landmark.movable = False
        world.landmark.size = 0.10
        #add obss
        world.obstacle = [Obstacle() for i in range(num_obstacle)]
        for i, o in enumerate(world.obstacle):
            o.name = 'obs %d' % i
            o.collide = True
            o.movable = True
            o.size = 0.10

        return world

    def reset_world(self, world):
        # random properties for agents

        # set random initial states
        #world.agent.state.p_pos = np.random.uniform([-1.9, -1.9], [-1.7, -1.7], world.dim_p)
        world.agent.state.p_pos = np.array([0.0,0.0])
        world.agent.state.v = 0.0
        world.agent.state.yaw = 0.0

        #world.landmark.state.p_pos = np.random.uniform([1.7,-1.9], [1.9,1.9], world.dim_p)
        world.landmark.state.p_pos = np.array([4.0,4.0])
        #for i,obs in enumerate(world.obstacle):
            #obs.state.p_pos = np.array([3.0,3.0])

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        rew += self.reward_goal_agnet(agent, world)
        rew += self.reward_arrived(agent, world)
        if agent.collide:
            for o in world.obstacle:
                if self.is_collision(o,agent):
                    rew -= 100
        return rew

    def reward_goal_agnet(self,agent,world):  
        rew=0
        dists_BeforeAction = np.sqrt(np.sum(np.square(agent.state.before_action_p_pos - world.landmark.state.p_pos)))
        dists_AfterAction = np.sqrt(np.sum(np.square(agent.state.p_pos - world.landmark.state.p_pos)))
        rew += (dists_BeforeAction-dists_AfterAction)/dists_BeforeAction
        return rew*30
    
    def reward_arrived(self,agent,world):
        rew=0
        if(self.arrive_target_region(agent,world.landmark)):
            rew +=100
        return rew

    #def observation(self, agent, world):
        #ob = []
        #for i,o in enumerate(world.obstacle):
            #ob.append(o.state.p_pos)
        #return np.concatenate([world.landmark.state.p_pos] + [agent.state.p_pos] +[[agent.state.v]] + [[agent.state.yaw]]+ ob)
    def observation(self, agent, world):
        return world.landmark.state.p_pos - agent.state.p_pos

    def done(self,agent,world):
        if(self.arrive_target_region(agent,world.landmark)):
            return True
        else:
            return False

    def arrive_target_region(self,agent,landmark):
        #所有的目标循环下,目前是一个
        delta_pos = agent.state.p_pos - landmark.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = (agent.size) + landmark.size
        return True if dist < dist_min else False