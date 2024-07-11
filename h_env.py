import numpy as np
import pybullet as p
import pybullet_data
import random
import math
import time
import pygame
from gymnasium.spaces import Box
from gymnasium import Env
import cv2

distance = 100000
img_w, img_h = 64, 64
class hEnv(Env):
    def __init__(self, use_controll=True, render=True):
        super(hEnv, self).__init__()
        self.use_controll = use_controll
        self.render = render
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.observation_space = Box(low=0, high=255, shape=(3, img_h, img_w), dtype=np.uint8)  # Changed shape to (3, ...)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.steps = 0
        self.obs = np.zeros((3, img_h, img_w), dtype=np.uint8)  # Changed shape to (3, ...)
        print(np.array(self.obs).shape)
        if self.use_controll:
            self.xbox()
        self.frame = 0
        if self.use_controll and pygame.joystick.get_count() == 0:
            print("No joystick")

    def xbox(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

    def load_model(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf", basePosition=[0, 0, 0])
        self.model = p.loadURDF("husky/husky.urdf", basePosition=[0, 0, 0])
        self.walls = p.loadURDF("D:/Proiecte/D3RLPY/D3RLPY/URDF/walls/walls.urdf", basePosition=[0, 0, 0])
        for i in range(p.getNumJoints(self.model)):
            print(p.getJointInfo(self.model,i))
        p.setGravity(0,0,-9.8)

    def step(self, action):
        reward, done = self.get_reward_done()
        obs = self.get_observation()

        p.setJointMotorControl2(self.model, 2, p.VELOCITY_CONTROL, targetVelocity=action[0]*-10)
        p.setJointMotorControl2(self.model, 3, p.VELOCITY_CONTROL, targetVelocity=action[1]*-10)
        p.setJointMotorControl2(self.model, 4, p.VELOCITY_CONTROL, targetVelocity=action[0]*-10)
        p.setJointMotorControl2(self.model, 5, p.VELOCITY_CONTROL, targetVelocity=action[1]*-10)
        p.stepSimulation()
        self.steps += 1
        info = {}
        truncated = False
        print(reward)
        return obs, reward, done, truncated, info

    def controll(self):
        if not self.use_controll:
            return [0, 0, 0, 0]

        pygame.event.get()
        left_joystick_value = pygame.joystick.Joystick(0).get_axis(1)
        right_joystick_value = pygame.joystick.Joystick(0).get_axis(3)

        print(left_joystick_value, right_joystick_value)

        output = [left_joystick_value, right_joystick_value]
        return output

    def make_photo(self):
        agent_pos, agent_orn = list(p.getLinkState(self.model, 8, computeForwardKinematics=True))[:2]
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        zA = zA + 0.3
        xB = xA + math.cos(yaw) * distance
        yB = yA + math.sin(yaw) * distance
        zB = zA
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[xA, yA, zA],
            cameraTargetPosition=[xB, yB, zB],
            cameraUpVector=[0, 0, 1.0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1.5, nearVal=0.02, farVal=25)
        self.rgb_image = p.getCameraImage(img_w, img_h, view_matrix, projection_matrix, shadow=True,
                                          renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # Reshape and extract first 3 channels (RGB)
        # self.rgb_image_temp = np.reshape(self.rgb_image[2], [img_h, img_w, 4])[:, :, :3].astype(np.uint8)

        self.rgb_image_temp = np.reshape(self.rgb_image[2], [4, img_h, img_w])[:3, :, :].astype(np.uint8)

        # Convert to grayscale
        # self.BW_image = cv2.cvtColor(self.rgb_image_temp, cv2.COLOR_RGB2GRAY)
        #
        # # Reshape to add channel dimension (single channel for grayscale)
        # self.BW_image = np.expand_dims(self.BW_image, axis=0)  # Add channel dimension
        #
        # # Convert to uint8 for consistency (optional)
        # self.BW_image = self.BW_image.astype(np.uint8)

        # Return grayscale image
        return self.rgb_image_temp

    def get_observation(self):
        obs = self.make_photo()
        return np.array(obs)  # Convert list of observations to a numpy array

    def spawn_target(self):
        random_x = random.uniform(-5.0,5.0)
        random_y = random.uniform(-5.0, 5.0)
        if random_x < 2 and random_x>0:random_x= 2
        if random_x >-2 and random_x<0:random_x=-2
        if random_y < 2 and random_y>0:random_y=2
        if random_y >-2 and random_y<0:random_y=-2
        base_position = [random_x, random_y, 0.5]
        self.target = p.loadURDF('D:/Proiecte/D3RLPY/D3RLPY/URDF/walls/cubeRed.urdf', basePosition = base_position)

    def num_to_range(self,num, inMin, inMax, outMin, outMax):
        return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin) * 1.0)

    def get_reward_done(self):
        joint_info = p.getBasePositionAndOrientation(self.model)
        x1,y1,_ = joint_info[0]

        base_position = p.getBasePositionAndOrientation(self.target)
        x2,y2,_ = base_position[0]

        distance = abs(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        done = False

        rosu = np.reshape(self.rgb_image[2], (img_h, img_w, 4))  # Changed from (img_h, img_w,4) to (img_h, img_w,3)
        rosu = rosu[:,:,:3].astype(np.uint8)  # Ensure it's only RGB without alpha channel
        cv2.cvtColor(rosu, cv2.COLOR_BGR2RGB)

        lower_red = np.array([100, 0, 0])
        upper_red = np.array([255, 0, 0])
        red_mask = cv2.inRange(rosu, lower_red, upper_red)
        rosu_cantitate = np.count_nonzero(red_mask)
        rosu_cantitate = self.num_to_range(rosu_cantitate, 0, img_h*img_w, 0, 5)

        reward = rosu_cantitate
        if rosu_cantitate < 1:
            rosu_cantitate = 0

        if distance < 1.25 and rosu_cantitate > 2:
            done = True
            reward = rosu_cantitate
        if self.steps >= 1250:
            done = True
            reward = rosu_cantitate
        return reward, done

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if p.isConnected():
            p.disconnect()
            time.sleep(1)
        if self.render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=7.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0, 0, 0])
        self.steps = 0
        self.load_model()
        self.spawn_target()
        if self.use_controll:
            self.xbox()
        obs = self.make_photo()
        info = {}
        return obs, info