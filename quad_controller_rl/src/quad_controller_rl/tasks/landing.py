"""Landing task."""

import numpy as np
from gym import spaces
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose, Twist, Wrench
from quad_controller_rl.tasks.base_task import BaseTask

class Landing(BaseTask):
    """More challenging task where the goal is to land on the ground softly."""

    def __init__(self):
        # State space: <position_x, .._y, .._z, orientation_x, .._y, .._z, .._w, velocity_x, .._y, .._z>
        cube_size = 300.0  # env is cube_size x cube_size x cube_size
        max_velocity = 25.0
        max_acceleration = 12.0
        self.observation_space = spaces.Box(
            np.array([- cube_size / 2, - cube_size / 2,       0.0, 
                      -1.0, -1.0, -1.0, -1.0, 
                      -max_velocity, -max_velocity, -max_velocity, 
                      -max_acceleration, -max_acceleration, -max_acceleration]),
            np.array([  cube_size / 2,   cube_size / 2, cube_size,  
                      1.0,  1.0,  1.0,  1.0, 
                      max_velocity, max_velocity, max_velocity, 
                      max_acceleration, max_acceleration, max_acceleration])
        )
        #print("Takeoff(): observation_space = {}".format(self.observation_space))  # [debug]

        # Action space: <force_x, .._y, .._z, torque_x, .._y, .._z>
        max_force = 25.0
        max_torque = 25.0
        self.action_space = spaces.Box(
            np.array([-max_force, -max_force, -max_force, -max_torque, -max_torque, -max_torque]),
            np.array([ max_force,  max_force,  max_force,  max_torque,  max_torque,  max_torque]))
        #print("Takeoff(): action_space = {}".format(self.action_space))  # [debug]

        # Task-specific parameters
        self.max_duration = 10.0  # secs
        self.max_error_position = 15.0  # distance units
        self.start_position = np.array([0.0, 0.0, 10.0])  # start on 5 units below the target position
        self.target_position = np.array([0.0, 0.0, 0.0])  # target position to hover at
        self.weight_position = 0.5
        self.target_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # target orientation quaternion (upright)
        self.weight_orientation = 0.1
        self.target_velocity = np.array([0.0, 0.0, -1.2])  # target velocity (ideally should land slowly)
        self.weight_velocity = 0.6
        self.target_acceleration = np.array([0.0, 0.0, 10.0])  # target acceleration (ideally should land smoothly)
        self.weight_acceleration = 0.2

    def reset(self):
        # Reset episode-specific variables
        self.last_timestamp = None
        self.last_position = None
        self.last_velocity = None
        self.last_acceleration = None
        # Return initial position
        p = self.start_position + np.random.normal(0.5, 0.1, size=3)  # slight random position around the target
        return Pose(
                position=Point(*p),  # drop off from a slight random height
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            ), Twist(
                linear=Vector3(0.0, 0.0, 0.0),
                angular=Vector3(0.0, 0.0, 0.0)
            )

    def update(self, timestamp, pose, angular_velocity, linear_acceleration):
        # Prepare state vector (pose, orientation, velocity only; ignore angular_velocity, linear_acceleration)
        position = np.array([pose.position.x, pose.position.y, pose.position.z])
        orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        if self.last_timestamp is None:
            velocity = np.array([0.0, 0.0, 0.0])
            acceleration = np.array([0.0, 0.0, 0.0])
        else:
            velocity = (position - self.last_position) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
            acceleration = (velocity - self.last_velocity) / max(timestamp - self.last_timestamp, 1e-03)  # prevent divide by zero
        state = np.concatenate([position, orientation, velocity, acceleration])
        self.last_timestamp = timestamp
        self.last_position = position
        self.last_velocity = velocity
        self.last_acceleration = acceleration

        error_position = np.linalg.norm(self.target_position - state[0:3])  # Euclidean distance from target position vector
        error_orientation = np.linalg.norm(self.target_orientation - state[3:7])  # Euclidean distance from target orientation quaternion (a better comparison may be needed)
        error_velocity = np.linalg.norm(self.target_velocity - state[7:10])  # Euclidean distance from target velocity vector
        error_acceleration = np.linalg.norm(self.target_acceleration - state[10:13])  # Euclidean distance from target acceleration vector

        # Compute reward / penalty and check if this episode is complete
        done = False
        #reward = -min(abs(self.target_z - pose.position.z), 20.0)  # reward = zero for matching target z, -ve as you go farther, upto -20
        reward = -(self.weight_position * error_position + 
                   self.weight_orientation * error_orientation + 
                   self.weight_velocity * error_velocity +
                   self.weight_acceleration * error_acceleration)
        if error_position > self.max_error_position:
            if pose.position.z > 10.5:
                reward -= 500.0  # extra big penalty for flying up
            else:
                reward -= 300.0  # extra penalty, agent strayed too far
            done = True
        # if pose.position.z <= 0.2 and error_velocity > 10.0:
        #     reward -= 20.0  # extra penalty for landing too fast
        #     done = True
        if timestamp > self.max_duration:  # agent has run out of time
            reward -= 300.0  # extra penalty, agent has taken too long
            done = True
        if pose.position.z == 0.0:
            reward += 100.0
            if error_velocity < 0.5:
                reward += 600.0
            elif error_velocity < 2.0:  # agent has successfully landed
                reward += 400.0  # extra reward for landing
            elif error_velocity < 5.0:  # smooth landing bonus
                reward += 100.0  # extra reward for landing
            else:
                reward -= 50.0  # too fast
            done = True

        # Take one RL step, passing in current state and reward, and obtain action
        # Note: The reward passed in here is the result of past action(s)
        action = self.agent.step(state, reward, done)  # note: action = <force; torque> vector

        # Convert to proper force command (a Wrench object) and return it
        if action is not None:
            action = np.clip(action.flatten(), self.action_space.low, self.action_space.high)  # flatten, clamp to action space limits
            return Wrench(
                    force=Vector3(action[0], action[1], action[2]),
                    torque=Vector3(action[3], action[4], action[5])
                ), done
        else:
            return Wrench(), done
