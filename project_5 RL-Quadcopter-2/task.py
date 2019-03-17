import numpy as np
import gym
from physics_sim import PhysicsSim
from scipy.spatial.distance import euclidean


class Task:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state


class Takeoff:
    """Takeoff (environment) that defines the goal and provides feedback to the agent. Model this like the OpenAI Gym
    environments for compatibility with the same agents."""

    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None, runtime=5., target_pos=None):
        """
        Params
        ======
            init_pose: init position (x,y,z) of the quadcopter and the Euler angles phi (roll), theta (pitch), psi (yaw)
            init_velocities: init velocity (vx, vy, vz) of the quadcopter
            init_angle_velocities: init angular velocities in rad/s for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_low = 0
        self.action_high = 900
        self.low_state = np.array([-150.0, -150.0, 0.0, 0.0, 0.0, 0.0])
        self.high_state = np.array([150.0, 150.0, 300.0, 2 * np.pi, 2 * np.pi, 2 * np.pi])
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.action_repeat = 1
        self.state_size = self.action_repeat * 6
        self.action_size = 4

        self.observation_space = gym.spaces.Box(low=np.repeat(self.low_state, 1),
                                                high=np.repeat(self.high_state, 1),
                                                dtype=np.float32)

        self.action_space = gym.spaces.Box(low=self.action_low,
                                           high=self.action_high,
                                           shape=(4,),
                                           dtype=np.float32)

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        delta_xy = euclidean(self.sim.pose[:2], self.target_pos[:2])
        max_delta_xy = euclidean(self.low_state[:2], self.high_state[:2])
        delta_z = self.sim.pose[2] - self.target_pos[2]
        max_delta_z = 300

        distance_penalty = (delta_xy / max_delta_xy) - 2 * (delta_z / max_delta_z)
        angular_penalty = (np.sum(self.sim.pose[3:]) / (6 * np.pi))
        velocity_reward = 2 * self.sim.v[2]
        velocity_penalty = abs(self.sim.v[1]) + abs(self.sim.v[0])

        reward = (1. - distance_penalty - angular_penalty + velocity_reward - velocity_penalty) / 10

        return reward

    def step(self, action):
        """Uses action (rotor speeds) to obtain next state, reward, done."""

        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(action)  # update the sim pose and velocities
            reward += self.get_reward()
            # Extra reward for reaching target position
            if self.sim.pose[2] >= self.target_pos[2]:
                reward += 100
                done = True
                # Extra reward/penalty for getting within +/- 5 of the target position in the xy plane
                if (self.sim.pose[0] - 5 < self.target_pos[0] < self.sim.pose[0] + 5) and (
                        self.sim.pose[1] - 5 < self.target_pos[1] < self.sim.pose[1] + 5):
                    reward += 100
            # Extra penalty for running out of time
            if self.sim.time > self.sim.runtime:
                reward -= 100
            # Extra penalty going out of bounds (crashing)
            if np.any(self.sim.pose[:3] <= self.low_state[:3]) or np.any(self.sim.pose[:3] > self.high_state[:3]):
                reward -= 100
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""

        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state

