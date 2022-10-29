import gym
from gym import spaces
import random
from stable_baselines import PPO2
import numpy as np
from math import sqrt,cos,sin,isnan
import tensorflow as tf
import pygame
import sys
from time import sleep
import subprocess
import matplotlib.pyplot as plt

global vision

# Clamp functions to keep the agent within the arena
def clampX(n):
    minn = -1874
    maxn = 1055
    return max(min(maxn, n), minn)

def clampY(n):
    minn = 2055
    maxn = 5005
    return max(min(maxn, n), minn)

# Fit x and y values to the screen of PyGame. This has no impact on what the actual
# agent sees and is purely visual
def fitX(n):
    return (n+1874)/3

def fitY(n):
    return (n-2055)/3

def calcDist(x, y, enemy):
    return sqrt((enemy[0] - x)**2 + (enemy[1] - y)**2 )

# Normalising x, y and distance values
def normalX(x):
    return 6.8282690337999E-4*x + 0.27961761693411

def normalY(y):
    return 6.7796610169492E-4*y - 2.3932203389831

def normalDist(dist):
    return 2.3809523809524E-4 * dist

def rotateAgent(currentRotation, deg):
    angle = np.deg2rad(deg)
    newRotation = (currentRotation[0]*cos(angle) - currentRotation[1]*sin(angle),currentRotation[0]*sin(angle) + currentRotation[1]*cos(angle))
    return (round(newRotation[0],2),round(newRotation[1],2))

def calcVision(agent,enemy,rotation):
    # Get the two vectors to compare and format them correctly
    correctRotation = [enemy[0]-agent[0],enemy[1]-agent[1]]
    actualRotation = [rotation[0],rotation[1]]

    # Calculate the angle between the vectors and convert that angle to degrees
    unit_vector_1 = correctRotation / np.linalg.norm(correctRotation)
    unit_vector_2 = actualRotation / np.linalg.norm(actualRotation)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)

    if isnan(np.rad2deg(angle)):
        return 0
    else:
        return np.rad2deg(angle)


class ShooterEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    global reward
    global location
    global goalReached
    global enemyLocation
    global distance
    global possibleLocations
    global frame
    global currentRotation
    global rightDirection
    global enemyHit
    global isShooting
    global shotPunish

    # A list to build a hit statistic and a boolean to track the first hit
    global hitStats
    global firstHit
    global statistics
    global statsLeft

    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)


    def __init__(self):
        super(ShooterEnv, self).__init__()

        # Initialize visualization
        pygame.init()
        pygame.display.set_caption('PPO2 Agent')
        # FPS controller
        self.fps_controller = pygame.time.Clock()
        self.game_window = pygame.display.set_mode((1000, 1000))

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1,shape=(4,))
        self.observation_space = spaces.Box(low=np.array([-1,-1,-1,-1,0,0,-1,-1,-1,-1,0,0]), high=np.array([1,1,1,1,1,1,1,1,1,1,1,1]),dtype=np.float32)

        #self.possibleLocations = ((-1440,4310),(630,3480))
        self.possibleLocations = ((-1440,4310),(630,3480),(-400,4310),(630,4310),(-1440,2340),(-400,2340),(630,2340),(-1440,3480))
        #self.location = (-540,3432)
        self.location = (-400,3500)
        self.enemyLocation = random.choice(self.possibleLocations)
        self.goalReached = False
        self.distance = calcDist(self.location[0], self.location[1], self.enemyLocation)
        self.currentRotation = (1,0)
        self.frame = 0
        self.reward = 0
        self.rightDirection = False
        self.enemyHit = False
        self.shotPunish = 0.1

        self.hitStats = [0] * 100
        self.firstHit = True
        self.statistics = True
        self.statsLeft = True


    def step(self, action):
        print(action)
        self.reward = 0
        # Switched to 100 steps for plotting purposes
        #if self.frame >= 200:
        if self.frame >= 100:
            self.frame = 0
            done = True
        else:
            self.frame +=1
            done = False
        x = clampX(self.location[0] + (action[0]*50))
        y = clampY(self.location[1] + (action[1]*50))
        self.location = (x,y)

        # Move the opponent
        xRan = random.uniform(-1,1)
        yRan = random.uniform(-1,1)
        xOP = clampX(self.enemyLocation[0] + (xRan*50))
        yOP = clampY(self.enemyLocation[1] + (yRan*50))
        self.enemyLocation = (xOP, yOP)

        # Rotate the agent
        self.currentRotation = rotateAgent(self.currentRotation, action[2]*15)

        self.distance = calcDist(x,y, self.enemyLocation)
        directionError = calcVision(self.location,self.enemyLocation,self.currentRotation)
        correctRotation = [self.enemyLocation[0]-self.location[0],self.enemyLocation[1]-self.location[1]]
        correctRotationNorm = correctRotation/np.linalg.norm(correctRotation)
        if isnan(correctRotationNorm[0]) or isnan(correctRotationNorm[1]):
            correctRotationNorm = self.currentRotation

        if action[3] > 0:
            self.isShooting = True
        else:
            self.isShooting = False
        self.checkReward(action)

        observation = np.array([normalX(self.location[0]), normalY(self.location[1]), normalX(self.enemyLocation[0]), normalY(self.enemyLocation[1]), float(self.goalReached),
                                normalDist(self.distance),self.currentRotation[0],self.currentRotation[1], correctRotationNorm[0], correctRotationNorm[1],
                                float(self.rightDirection),float(self.enemyHit)])

        info = {}
    
        return observation, self.reward, done, info

    def reset(self):
        #self.possibleLocations = ((-1440,4310),(630,3480))
        self.possibleLocations = ((-1440,4310),(630,3480),(-400,4310),(630,4310),(-1440,2340),(-400,2340),(630,2340),(-1440,3480))
        self.enemyLocation = random.choice(self.possibleLocations)
        self.goalReached = False
        self.rightDirection = False
        self.enemyHit = False
        self.isShooting = False
        self.shotPunish = 0.1

        # Reset the hit variable for the statistic
        self.firstHit = True

        # Reset agent to an entirely random location
        x = random.randint(-1874,1055)
        y = random.randint(2055,5005)
        self.location = (x,y)
        self.currentRotation = (random.uniform(-1,1),random.uniform(-1,1))

        # Fixing the agents position for statistical purposes
        """
        if self.statistics:
            self.location = (-540,3432)
            self.currentRotation = (0,1)
            if self.statsLeft:
                self.enemyLocation = (-1850, 3500)
            else:
                self.enemyLocation = (1050, 3500)
        """

        directionError = calcVision(self.location,self.enemyLocation,self.currentRotation)
        self.distance = calcDist(self.location[0], self.location[1], self.enemyLocation)
        self.frame = 0
        self.reward = 0

        correctRotation = [self.enemyLocation[0]-self.location[0],self.enemyLocation[1]-self.location[1]]
        correctRotationNorm = correctRotation/np.linalg.norm(correctRotation)
        if isnan(correctRotationNorm[0]) or isnan(correctRotationNorm[1]):
            correctRotationNorm = self.currentRotation

        # Calculating the new reward based on the new enemy position using a dummy action
        self.checkReward((0,0,0,0))

        observation = np.array([normalX(self.location[0]), normalY(self.location[1]), normalX(self.enemyLocation[0]), normalY(self.enemyLocation[1]), float(self.goalReached),
                                normalDist(self.distance),self.currentRotation[0],self.currentRotation[1], correctRotationNorm[0], correctRotationNorm[1],
                                float(self.rightDirection),float(self.enemyHit)])
        return observation

    def render(self, mode='human'):
        self.game_window.fill(self.black)
        x1 = fitX(self.location[0])
        y1 = fitY(self.location[1])
        x2 = fitX(self.enemyLocation[0])
        y2 = fitY(self.enemyLocation[1])
        pygame.draw.rect(self.game_window, self.green, pygame.Rect(int(x1), int(y1), 10, 10))
        pygame.draw.rect(self.game_window, self.red, pygame.Rect(int(x2), int(y2), 10, 10))
        rotationDirection = (x1+20*self.currentRotation[0],y1+20*self.currentRotation[1])
        pygame.draw.line(self.game_window, self.white,(x1+5,y1+5),rotationDirection,5)

        # Draw the agents shot
        if self.isShooting:
            rotationDirection1 = (x1+1000*self.currentRotation[0],y1+1000*self.currentRotation[1])
            pygame.draw.line(self.game_window, self.blue,(x1,y1),rotationDirection1,10)
        self.showScore()
        pygame.display.update()
       
    def showScore(self):
        score_font = pygame.font.SysFont('consolas', 20)
        score_surface = score_font.render('Reward: ' + str(round(self.reward,2)), True, self.white)
        score_rect = score_surface.get_rect()
        score_rect.midtop = (1000 / 10 + 50, 15)

        #dist_surface = score_font.render('Distance: ' + str(self.distance), True, self.white)
        dist_surface = score_font.render('Offset: ' + str(round(calcVision(self.location,self.enemyLocation,self.currentRotation),2)), True, self.white)
        dist_rect = dist_surface.get_rect()
        dist_rect.midtop = (1000 / 10 + 50, 35)

        hit_surface = score_font.render('Enemy hit: ' + str(self.enemyHit), True, self.white)
        hit_rect = hit_surface.get_rect()
        hit_rect.midtop = (1000 / 10 + 50, 55)
        
        self.game_window.blit(score_surface, score_rect)
        self.game_window.blit(dist_surface, dist_rect)
        self.game_window.blit(hit_surface, hit_rect)

    def checkReward(self, action):
        directionError = calcVision(self.location,self.enemyLocation,self.currentRotation)
        self.reward = 0

        correctRotation = [self.enemyLocation[0]-self.location[0],self.enemyLocation[1]-self.location[1]]
        correctRotationNorm = correctRotation/np.linalg.norm(correctRotation)
        if isnan(correctRotationNorm[0]) or isnan(correctRotationNorm[1]):
            correctRotationNorm = self.currentRotation

        # Calculating the new reward based on the new enemy position
        # Reward or punish vision
        if directionError <= 5:
            #self.reward += 0.5
            self.rightDirection = True

        else:
            self.rightDirection = False
        #self.reward += (-0.0027777777777778*directionError)/2
        if directionError <= 0.5:
            self.reward += 0
        else:
            self.reward -= (0.1099*np.log(24.8396*directionError))/2

        # Reward or punish distance to enemy
        if self.distance <= 750:
            self.goalReached == True
            self.reward += 0.5

        else:
            self.goalReached == False
            #self.reward += -((self.distance/100)**2)
            self.reward += (-0.00025*self.distance+0.05)/2

        # Reward or punish shooting
        if self.isShooting:
            if self.rightDirection:
                self.reward += 0.5
                self.enemyHit = True
                self.shotPunish = 0.1

                if self.firstHit:
                    self.firstHit = False
                    self.hitStats[self.frame] += 1
            else:
                if self.vision:
                    self.reward += -self.shotPunish
                    self.shotPunish += 0.1
                self.enemyHit = False

        for act in action:
            self.reward += -0.1*(act**2)


    def close (self):
        pass

env = ShooterEnv()

#policyArg = dict(act_fun = tf.nn.tanh, layers = [(64,64)])
policyArg = dict(act_fun = tf.nn.tanh)

# Names of older iteraions to be conveniently used if needed
#name = "350k_always_negative"
#name = "350k_base"
#name = "30kk_remote"
#name = "5kk_punishment"
#name = "5kk_default"
#name = "5kk_punishment"
#name = "350k_punish"
#name = "350k_done"
#name = "350k_done2"
#name = "350k_done_negative"
#name = "350k_no_goal"
#name = "350k_normal"
#name = "350k_normal_square"
#name = "350k_normal_better"
#name = "350k_no_sqrt"
#name = "350k_eight_spawns"
#name = "350k_eight_spawns_reset"
#name = "20kk_eight_spawns_reset"

name = "3kk_increasing_punish"

# Set to True for training purposes and False for evaluating an already trained model
if False:
    model = PPO2('MlpPolicy', env, gamma=0.99, n_steps=128, ent_coef=0.01, learning_rate=0.00025, vf_coef=0.5, max_grad_norm=0.5,
                 lam=0.95, nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_vf=None, verbose=1, tensorboard_log="./" + name + "_log/",
                 _init_setup_model=True, policy_kwargs=policyArg, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)


    # Learn in two steps. First without punishing missing the enemy
    #env.vision = False
    #model.learn(total_timesteps=500000)
    env.vision = True
    #env.reset()
    model.learn(total_timesteps=3000000)
    model.save(name)
else:
    model = PPO2.load(name)

obs = env.reset()
env.location = (-400,3500)
env.vision = True

for i in range(10000):
    pygame.event.get()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    sleep(0.075)
    if done:
        obs = env.reset()


# This section only serves to build statistics concering the agents behaviour
""" 
# Enabling tracking of the statistics
env.statistics = True
env.statsLeft = True
for i in range(10000):
    pygame.event.get()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # Disable sleep timer to create statistic more quickly
    #sleep(0.075)
    sleep(5)
    if done:
        obs = env.reset()


# Plot the collected values
xAxis = list(range(1,101))
yAxisLeft = env.hitStats

env.statsLeft = False
env.hitStats = [0] * 100
for i in range(10000):
    pygame.event.get()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    # Disable sleep timer to create statistic more quickly
    #sleep(0.075)
    if done:
        obs = env.reset()

yAxisRight = env.hitStats
plt.bar([x - 0.2 for x in xAxis], yAxisLeft, 0.4, label = 'Opponent on the left')
plt.bar([x + 0.2 for x in xAxis], yAxisRight, 0.4, label = 'Opponent on the right')

plt.xlabel("Steps taken until the first shot was hit")
plt.ylabel("Episodes")
plt.legend()
plt.show()
plt.savefig('hitStats.png')
"""

#bashCommand = "tensorboard --logdir \"./no_boal_log/PPO2_1/\""
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()


