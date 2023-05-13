import numpy as np
import math

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

class AgentState(EntityState):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        self.before_action_p_pos=None#更新动作前的位置
        self.before_action_p_vel=None#更新动作后的位置
        self.v = None
        self.yaw = None

class AgentAction:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None

class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # color
        self.color = None
        # state
        self.state = EntityState()

class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()
        self.color = np.array([0.25, 0.25, 0.25])

class Obstacle(Entity):
    def __init__(self):
        super().__init__()
        self.color = np.array(([0.75,0.85,0.25]))

class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        self.color = np.array([0.35, 0.35, 0.85])
        # control range
        self.amax = 1.0
        self.wmax = 1.0
        self.u_range = 1.0
        self.u_noise = None
        # state
        self.state = AgentState()
        # action
        self.action = AgentAction()

class World:
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agent = []
        self.landmark = []
        self.obstacle = []
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1

    # return all entities in the world
    @property
    def entities(self):
        return [self.agent] + [self.landmark]

    # update state of the world
    def step(self):
        #update agent action/state
        control = self.apply_noise_into_control(False)  # True为有噪音,false为没有噪音,默认设置false
        self.integrate_state(control)

    def apply_noise_into_control(self,noise_flag=False):
        if noise_flag is True:
            noise = np.random.randn(*self.agent.action.u.shape) * self.agent.u_noise if self.agent.u_noise else 0.0
            control = self.agent.action.u + noise
        else:
            control = self.agent.action.u
        return control

    # integrate physical state
    def integrate_state(self,control):
        self.agent.state.before_action_p_pos=(self.agent.state.p_pos).copy()
        if (control is not None):
            self.control_vel_yaw(self.agent,control)

    def uav_model(self,agent,dt,a,w):
        #获取状态
        x0=agent.state.p_pos[0]
        y0=agent.state.p_pos[1]
        #v0 = agent.state.v
        yaw0 = agent.state.yaw

        #运动学模型
        #xf = x0 + v0*dt*math.cos(yaw0 + w*dt/2)
        #yf = y0 + v0*dt*math.sin(yaw0 + w*dt/2)
        #vf = v0 + a*dt
        #yawf = yaw0 + w*dt
        xf = x0+a
        yf = y0 + w

        #更新状态
        agent.state.p_pos[0]=xf
        agent.state.p_pos[1]=yf
        #agent.state.v = vf
        #agent.state.yaw = yawf

    def control_vel_yaw(self,agent,control):

        #control[0] = math.tanh(control[0])
        #control[1] = math.tanh(control[1])

        a_low = -agent.amax
        a_high = agent.amax
        a = a_low + (control[0]+1)/2*(a_high - a_low)

        w_low = -agent.wmax
        w_high = agent.wmax
        w = w_low + (control[1]+1)/2 * (w_high - w_low)

        self.uav_model(agent,self.dt,a,w)
