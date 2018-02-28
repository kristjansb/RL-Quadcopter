"""Takeoff task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask
import os

from quad_controller_rl import util

class Suck(BaseTask):
    """Simple task where the goal is to lift off the ground and reach a target height."""

    def __init__(self):
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, -1.0, -1.0, -1.0, -1.0, -np.inf, -np.inf, -np.inf]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  1.0,  1.0,  1.0,  1.0, np.inf, np.inf, np.inf]))
        
        max_force = 25
        max_torque = 0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        
        
        self.phase = -1
        
        self.desc = ["Determine Fg = m*g", "Determine Z-Drag", "Determine X-Drag", "DONE"]
        
        self.base_filename = util.get_param('out')
        
        
        self.Fg = 19.62


    def reset(self):
        self.phase += 1
        print("Phase {0}: {1}".format(self.phase, self.desc[self.phase]))
        
        self.last_position = None
        self.last_velocity = None
        self.last_timestep = None
        
        self.alpha = 1
        self.gamma = 0.999
        
        self.done_counter = 0
        self.last_position = None
        self.last_timestep = None
        self.positions = np.zeros([600, 2])
        
        if (self.phase == 0):
            return Pose(
                    position=Point(0.0, 0.0, 200),  # drop off from a slight random height
                    orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                ), Twist(
                    linear=Vector3(0.0, 0.0, 0.3),
                    angular=Vector3(0.0, 0.0, 0.0)
                )
        elif (self.phase == 1):
            return Pose(
                    position=Point(0.0, 0.0, 1),  # drop off from a slight random height
                    orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                ), Twist(
                    linear=Vector3(0.0, 0.0, 20),
                    angular=Vector3(0.0, 0.0, 0.0)
                )
        elif (self.phase == 2):
            return Pose(
                    position=Point(-120, 0.0, 100),  # drop off from a slight random height
                    orientation=Quaternion(0.0, 0.0, 0.0, 0.0),
                ), Twist(
                    linear=Vector3(20, 0.0, 5),
                    angular=Vector3(0.0, 0.0, 0.0)
                )
        else:
            raise ValueError("Unknown Phase {}".format(self.phase))

    def update(self, timestamp, pose, av, linear_acceleration):
        # Prepare state vector (pose only; ignore angular_velocity, linear_acceleration)
        
#         print ("======================")
#         print (timestamp)
#         print (pose)
#         print (av)
#         print (linear_acceleration)
        
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        
        action = np.zeros(3)
        action[2] = self.Fg
        done = False
        
        if (self.last_position is not None):
            delta = (pos - self.last_position)
        else:
            delta = np.zeros(3)
        
        if (self.phase == 0):
            
            self.Fg -= self.alpha*delta[2]
            self.alpha *= self.gamma

            if (delta[2] < 0.0000001):
                self.done_counter += 1
            else:
                self.done_counter = 0

            action[2] = self.Fg

            if (self.done_counter > 300):
                done = True
                print ("Fg = {}".format(self.Fg))
                    
        elif (self.phase == 1):
            self.positions[self.done_counter, 0] = timestamp
            self.positions[self.done_counter, 1] = pos[2]
            
            self.done_counter += 1
            
            if (self.done_counter >= self.positions.shape[0]):
                done = True
                
                # Process position info
                PMAP = self.positions
                VMAP = np.zeros([PMAP.shape[0]-14, 2])
                for i in range(7, VMAP.shape[0]):
                    VMAP[i,0] = PMAP[i-7, 0]
                    VMAP[i,1] = (PMAP[i+7, 1]-PMAP[i-7,1])/(PMAP[i+7,0]-PMAP[i-7,0])
                    
                CMAP = np.zeros([VMAP.shape[0]-90])
                for i in range(CMAP.shape[0]):
                    t = VMAP[i+90,0] - VMAP[i,0]
                    v0 = VMAP[i   , 1]
                    vf = VMAP[i+90, 1]
                    CMAP[i] = (v0-vf)/(v0*vf*t+0.01)
                
                print ("Cf_z = {0} +/- {1}".format(np.mean(CMAP[30:]), np.std(CMAP[30:])))
                
        elif (self.phase == 2):
            self.positions[self.done_counter, 0] = timestamp
            self.positions[self.done_counter, 1] = pos[2]
            
            self.done_counter += 1
            
            if (self.done_counter >= self.positions.shape[0]):
                done = True
                
                # Process position info
                PMAP = self.positions
                VMAP = np.zeros([PMAP.shape[0]-14, 2])
                for i in range(7, VMAP.shape[0]):
                    VMAP[i,0] = PMAP[i-7, 0]
                    VMAP[i,1] = (PMAP[i+7, 1]-PMAP[i-7,1])/(PMAP[i+7,0]-PMAP[i-7,0])
                    
                CMAP = np.zeros([VMAP.shape[0]-90])
                for i in range(CMAP.shape[0]):
                    t = VMAP[i+90,0] - VMAP[i,0]
                    v0 = VMAP[i   , 1]
                    vf = VMAP[i+90, 1]
                    CMAP[i] = (v0-vf)/(v0*vf*t+0.01)
                
                print ("Cf_x = {0} +/- {1}".format(np.mean(CMAP[30:]), np.std(CMAP[30:])))
            
            
        else:
            raise ValueError("Unknown Phase {}".format(self.phase))
        
        self.last_position = pos
        

        return Wrench(
                force=Vector3(action[0], action[1], action[2]),
                torque=Vector3(0, 0, 0)
            ), done
