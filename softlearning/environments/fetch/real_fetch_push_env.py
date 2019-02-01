import abc
import os.path as osp
from collections import OrderedDict
import numpy as np
from serializable import Serializable
import gym
from gym.spaces import Box, Dict
from PIL import Image
import pickle
import tensorflow as tf

from softlearning.misc.utils import PROJECT_PATH

from softlearning.environments.fetch.push_rl_moveit_wrapper import *
from geometry_msgs.msg import TwistStamped,PoseStamped
from std_msgs.msg import Header
import geometry_msgs.msg

from joint_listener.srv import ReturnJointStates, ReturnEEPose
import rospy
import math
import time
from softlearning.environments.fetch.simple_camera import *

scale_control = 10

#INVERSE MODEL
latent_meta_path = '/scr/glebs/dev/softlearning/inverse_models/fetch_pushing_inverse_multi_step_lstm_unit_64_fwd_consist_28-01-2019_17-21-58/models/model_plan_test_99000.meta'
#UPN MODEL
#latent_meta_path = '/scr/glebs/dev/softlearning/upn_models/model_plan_test_7000.meta'

class FetchPushVisionEnv(gym.Env, Serializable, metaclass=abc.ABCMeta):
    """Implements the real fetch pushing environment with visual rewards"""

    def __init__(self, vision=True, random_init=False, use_latent=True, use_imitation=False, 
                random_action_horizon=0, fixed_goal=True, camera_port=0):

        #SETTING UP OBS SPACE AND AC SPACE

        #3 for ee pose and 3 for goal pos

        #3 for ee pose 128 for latent metric
        obs_high = np.inf*np.ones(3)
        self.observation_space = Box(-obs_high, obs_high, dtype=np.float32)

        #just x,y velocity
        ac_high = np.ones(2)
        self.action_space = Box(low=-ac_high, high=ac_high, dtype=np.float32)

        #MOVEIT
        rospy.init_node('fetch_sac_rl')
        self.moveit = RLMoveIt()
       

        #CAMERA
        self.camera = SimpleCamera(camera_port)

        self.curr_path_length = 0
        self.max_path_length = 15
        self.vision = vision
        self.random_init = random_init
        self.fixed_goal = fixed_goal
        self.use_latent = use_latent
        if use_imitation:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.latent_graph = tf.Graph()
            self.latent_sess = tf.Session(config=tf_config, graph=self.latent_graph)
            with self.latent_graph.as_default():
                saver = tf.train.import_meta_graph(latent_meta_path)
                saver.restore(self.latent_sess, latent_meta_path[:-5])
                self.latent_feed_dict = {
                    'ot': self.latent_graph.get_tensor_by_name('ot:0'),
                    'qt': self.latent_graph.get_tensor_by_name('qt:0'),
                    'plan': self.latent_graph.get_tensor_by_name('plan:0'),
                }
        if use_latent:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            self.latent_graph = tf.Graph()
            self.latent_sess = tf.Session(config=tf_config, graph=self.latent_graph)
            with self.latent_graph.as_default():
                saver = tf.train.import_meta_graph(latent_meta_path)
                saver.restore(self.latent_sess, latent_meta_path[:-5])
                self.latent_feed_dict = {
                    'ot': self.latent_graph.get_tensor_by_name('ot:0'),
                    'qt': self.latent_graph.get_tensor_by_name('qt:0'),
                    'og': self.latent_graph.get_tensor_by_name('og:0'),
                    'xt': self.latent_graph.get_tensor_by_name('xt:0'),
                    'xg': self.latent_graph.get_tensor_by_name('xg:0'),
                    'eff_horizons': self.latent_graph.get_tensor_by_name('eff_horizons:0'),
                    'atT_original': self.latent_graph.get_tensor_by_name('atT_0:0'),
                    'plan': self.latent_graph.get_tensor_by_name('plan:0'),
                }
                self.xg = self.latent_sess.run(self.latent_feed_dict['xg'],
                                                feed_dict={self.latent_feed_dict['og']: np.zeros((1, 100, 100, 3))})
        # if scale_and_bias_path is not None:
        #     with open(scale_and_bias_path, 'rb') as f:
        #         data = pickle.load(f)
        #         self.scale = data['scale']
        #         self.bias = data['bias']

        self.scale = 0
        self.bias = 0

        self.imgs = []
        self.full_imgs = []

        self.c_im_seq = []
        self.c_full_im_seq = []
        self.num_reset_calls = 0
        #Finally we want to make a fixed goal pose
        self.fixed_goal_array = [0.74,0.15,0.98]
        self.fixed_goal = self.moveit.fixed_pose(self.fixed_goal_array[0], self.fixed_goal_array[1])

        self.make_goal(self.fixed_goal)
        image_stats = [self.goal_img, self.fixed_goal_array]
        #pickle.dump(image_stats, open('/scr/glebs/dev/softlearning/goal_info/push_goal_info_1.pkl', 'wb'))
        #image = Image.fromarray(self.goal_img, 'RGB')
        #image.save('/scr/glebs/dev/softlearning/goal_info/push_goal_image_1.png')
        # self.quick_init(locals())
        image = Image.fromarray(self.goal_img, 'RGB')
        image.save('/scr/glebs/dev/softlearning/final_rollouts/push/inverse/goal_0.png')

        h_image = Image.fromarray(self.high_res_goal, 'RGB')
        h_image.save('/scr/glebs/dev/softlearning/final_rollouts/push/inverse/high_res_goal_0.png')
        self._Serializable__initialize(locals())

    def get_latent_metric(self, ot, qt, og=None):
        d = 0.85#1.0
        with self.latent_graph.as_default():
            # xt = self.latent_sess.run(self.latent_feed_dict['xt'],
            #                               feed_dict={self.latent_feed_dict['ot']: ot / 255.0,
            #                               self.latent_feed_dict['qt']:np.expand_dims(qt, axis=0)})
            # just use the latent representation without joint embedding seems to work a lot better!
            xt = self.latent_sess.run(self.latent_feed_dict['xg'],
                                            feed_dict={self.latent_feed_dict['og']: ot / 255.0})
            if og is None:
                xg = self.xg
            else:
                xg = self.latent_sess.run(self.latent_feed_dict['xg'],
                                                feed_dict={self.latent_feed_dict['og']: og / 255.0})
            error = np.abs(xt - xg)
            mask = (error > 1)
            dist = np.mean(np.sum(mask * (0.5 * (d**2) + d * (error - d)) + (1 - mask) * 0.5 * (error**2), axis=1))
            # dist = np.mean(mask * (0.5 * (d**2) + d * (error - d)) + (1 - mask) * 0.5 * (error**2))
        return dist

    def get_bc_action(self, ot, qt):
        if len(ot.shape) == 3:
            T = 1
        elif len(ot.shape) == 4:
            T = ot.shape[0]
        else:
            T = ot.shape[1]
        with self.latent_graph.as_default():
            plan = self.latent_sess.run(self.latent_feed_dict['plan'],
                                            feed_dict={self.latent_feed_dict['ot']: ot.reshape(1, T, 100, 100, 3) / 255.0,
                                                        self.latent_feed_dict['qt']:qt.reshape(1, T, 4)})
        return np.squeeze(plan)
#NOTUSED
    def get_plan(self, ot, qt, og=None):
        if og is None:
            og = self.goal_img
        eff_horizons = np.array([39])
        atT_original = np.random.uniform(-1., 1., size=(1, 39, self.model.nu))
        with self.latent_graph.as_default():
            plan = self.latent_sess.run(self.latent_feed_dict['plan'],
                                            feed_dict={self.latent_feed_dict['ot']:np.expand_dims(ot / 255.0, axis=0),
                                                        self.latent_feed_dict['qt']:np.expand_dims(qt, axis=0),
                                                        self.latent_feed_dict['og']:np.expand_dims(og / 255.0, axis=0),
                                                        self.latent_feed_dict['eff_horizons']:eff_horizons,
                                                        self.latent_feed_dict['atT_original']:atT_original})
        return np.squeeze(plan)

    def step(self, action):
        #print("Taking next step")
        img, qt, full_img = self.get_current_image_obs()
        #print("Got image")
        self.c_im_seq.append(img)
        self.c_full_im_seq.append(full_img)

        if hasattr(self, "goal_img"):
            vec= img / 255.0 - self.goal_img / 255.0
        else:
            vec = img / 255.0

        #actual_vec = self.get_actual_distance()
        if not self.use_latent:
            reward_dist = - np.linalg.norm(vec)
        else:
            reward_dist = -0.2*self.get_latent_metric(np.expand_dims(img, axis=0), qt.dot(self.scale) + self.bias)
            reward_dist = np.exp(reward_dist)
        #reward_ctrl = - np.square(action).sum()
        #actual_dist = np.linalg.norm(actual_vec)
        if self.vision and not self.use_latent:
            reward = reward_dist
        elif not self.use_latent:
            reward = reward_dist
        else:
            reward = reward_dist# + 0.1*reward_ctrl
        #print("GOT REWARD")
        self.take_action(action)
        #print("TOOK ACTION")
        ob = self._get_obs()
        #print("GOT OBS")
        self.curr_path_length +=1
        if self.curr_path_length == self.max_path_length:
            print("Max path reached")
            self.imgs.append(self.c_im_seq)
            self.full_imgs.append(self.c_full_im_seq)
            self.num_reset_calls += 1

            #print(self.num_reset_calls)
            if self.num_reset_calls == 3:
                print("Saving images to video")
                self.save_images_to_video()

            self.c_im_seq = []
            self.c_full_im_seq = []

            self.curr_path_length = 0
            done = True
        else:
            done = False

        return ob, reward, done, dict(reward_dist=reward_dist)

    def save_images_to_video(self):
        import imageio
        for i in range(3):
            print("/scr/glebs/dev/softlearning/final_rollouts/push/inverse/goal_0_%d.gif" % i)
            imageio.mimwrite("/scr/glebs/dev/softlearning/final_rollouts/push/inverse/goal_0_%d.gif" % i, np.array(self.imgs[i]))
            imageio.mimwrite("/scr/glebs/dev/softlearning/final_rollouts/push/inverse/full_goal_0_%d.gif" % i, np.array(self.full_imgs[i]))
    #return arm back to the fixed start position
    def reset(self):
        #while self.moveit.not_at_start():
        self.moveit.go_to_start()

        return self._get_obs()

    def _get_obs(self):

        pose = self.get_ee_pose()
        pose_array = [pose.position.x,pose.position.y,pose.position.z]
        #pose_array = [0,0,0]
        #current_img, _ = self.get_current_image_obs()
        #xo = self.latent_sess.run(self.latent_feed_dict['xg'],
        #                                               feed_dict={self.latent_feed_dict['og']: np.expand_dims(current_img / 255.0, axis=0)})
        
        if self.use_latent:
            return np.concatenate([
                pose_array,
                #ACTUAL GOAL
                #self.fixed_goal_array,
                #LATENT REP OF GOAL
                #np.squeeze(self.xg),
                #LATENT REP OF CURRENT OBS
                #np.squeeze(xo)


            ])
        return 0

    def get_current_image_obs(self):
        img, full_img = self.camera.capture()
        pose = self.get_ee_pose()

        pose_array = np.array([pose.position.x,pose.position.y,pose.position.z])
        #pose_array = np.array([0,0,0])
        return img, pose_array, full_img

    def get_goal_image(self):
        assert hasattr(self, 'goal_img')
        return self.goal_img

    def get_joint_info(self):
        rospy.wait_for_service("return_joint_states")
        try:
            s = rospy.ServiceProxy("/return_joint_states", ReturnJointStates)
            resp = s(joint_names)
        except rospy.ServiceException:
            print("error when calling return_joint_states: %s")
            sys.exit(1)
        for (ind, joint_name) in enumerate(joint_names):
            if(not resp.found[ind]):
                print("joint %s not found!"%joint_name)
        return np.array(resp.position)


    def get_ee_pose(self):
        #return 0;
        return self.moveit.get_ee_pose()
    def take_action(self, action):

        current_pose = self.get_ee_pose()

        #print(current_pose)
        #print(self.moveit.lower_point)
        z_command = self.moveit.height - current_pose.position.z
        if current_pose.position.x <= self.moveit.lower_point.position.x:
            print("Exceeds x lower")
            self.moveit.send_velocity(scale_control, 0, z_command)
        elif current_pose.position.y <= self.moveit.lower_point.position.y:
            print("Exceeds y lower")
            self.moveit.send_velocity(0, scale_control, z_command)
        elif current_pose.position.x >= self.moveit.upper_point.position.x:
            print("Exceeds x upper")
            self.moveit.send_velocity(-scale_control, 0, z_command)
        elif current_pose.position.y >= self.moveit.upper_point.position.y:
            print("Exceeds y upper")
            self.moveit.send_velocity(0,-scale_control,z_command)
        else:
            self.moveit.send_velocity(scale_control*action[0], scale_control*action[1], z_command)
        # while self.moveit.orientation_violated():
        #     self.moveit.fix_orientation()
        # self.moveit.send_velocity(0, 0, 0)

        # command = self.new_twist_command()
        # self.pub.publish(command)
        # self.rate.sleep()

    # def new_twist_command(self,x=0, y=0, z=0):
    #     twist = TwistStamped()

    #     twist.header.frame_id = 'base_link'
    #     twist.twist.linear.x = x
    #     twist.twist.linear.y = y
    #     twist.twist.linear.z = z

    #     return twist

    #sets goal, moves to that goal, and get's all info (image, joints) at the goal state
    def make_goal(self, pose):
        
        self.goal_pose = pose
        self.moveit.go_to(pose)

        self.goal_img, self.high_res_goal = self.camera.capture()

        self.xg = self.latent_sess.run(self.latent_feed_dict['xg'],
                                                        feed_dict={self.latent_feed_dict['og']: np.expand_dims(self.goal_img / 255.0, axis=0)})
