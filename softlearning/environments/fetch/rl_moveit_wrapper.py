#THIS IS BASICALLY GOING TO BE ONE MASSIVE WRAPPER THAT SENDS ALL PLANNING REQUESTS AND WHAT NOT TO THE ACTUAL MOVEIT COMMANDER RUNNING FROM 2.7

import time
import sys
import numpy as np
import geometry_msgs.msg
from geometry_msgs.msg import Point,Pose, Quaternion
from std_msgs.msg import String


#AS LONG AS THESE THINGS WORK, WE ARE IN BUSINESS!
import roslib
import rospy
roslib.load_manifest('joint_listener')
from joint_listener.srv import ReturnJointStates, ReturnEEPose, ReturnQuat, ReturnEuler, SendMoveitPose
import copy
import numpy.random as npr


joint_names = ["torso_lift_joint", "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]


#because tf2 doesn't play nicely with python3, we have to run both converters on the fetch and pipe over services
#just going to overload eepose for now because it's too much to make a new service
def quaternion_from_euler(x,y,z):
    rospy.wait_for_service("return_quat_from_euler")
    try:
        s = rospy.ServiceProxy("/return_quat_from_euler", ReturnQuat)
        resp = s(x,y,z)
    except rospy.ServiceException:
        print("error when calling return_quat_from_euler: %s")
        sys.exit(1)

    return [resp.x, resp.y, resp.z, resp.w]

def euler_from_quaternion(x,y,z,w):
    rospy.wait_for_service("return_euler_from_quat")
    try:
        s = rospy.ServiceProxy("/return_euler_from_quat", ReturnEuler)
        resp = s(x,y,z,w)
    except rospy.ServiceException:
        print("error when calling return_euler_from_quat: %s")
        sys.exit(1)

    return [resp.x, resp.y, resp.z]

quat = quaternion_from_euler(0, 1.5707, 1.5707)
upright = Quaternion(quat[0],quat[1],quat[2],quat[3])


f_lower_x = 0.4
f_upper_x = 0.7

f_lower_y = -0.35
f_upper_y = 0.15


f_height=0.96

f_start_x = 0.5
f_start_y = -0.1


class RLMoveIt(object):
    def __init__(self):

        self.lower_point = None
        self.upper_point = None
        self.lower_opp_point = None
        self.upper_opp_point = None
        self.start_point = None


        self.lower_x = None
        self.lower_y = None
        self.upper_x = None
        self.upper_y = None
        self.height = None

        self.get_start_point()
        self.get_ee_bounds()

    def random_pose(self):
        x = npr.uniform(low=self.lower_x, high=self.upper_x)
        y = npr.uniform(low=self.lower_y, high=self.upper_y)
        pose = Point(x, y, self.height)
        pose = Pose(position=pose, orientation=upright)

        return pose

    def get_ee_pose(self):
        rospy.wait_for_service("return_ee_pose")
        try:
            s = rospy.ServiceProxy("/return_ee_pose", ReturnEEPose)
            resp = s(joint_names)
        except rospy.ServiceException:
            print("error when calling return_ee_pose: %s")
            sys.exit(1)

        return resp.pose.pose

    def send_pose(self, pose):
        rospy.wait_for_service("send_moveit_pose")
        try:
            s = rospy.ServiceProxy("/send_moveit_pose", SendMoveitPose)
            resp = s(pose)
        except rospy.ServiceException:
            print("error when calling send_moveit_pose: %s")
            sys.exit(1)
        return resp

    def get_ee_bounds(self, fixed=True):
        if fixed:
            lower_x = f_lower_x
            lower_y = f_lower_y
            upper_x = f_upper_x
            upper_y = f_upper_y
            height = f_height

            self.lower_x = f_lower_x
            self.lower_y = f_lower_y
            self.upper_x = f_upper_x
            self.upper_y = f_upper_y
            self.height = f_height
        else:
            try:
              input("Please set the end effector to the first corner of your bounding rectangle, then click ENTER")
            except:
              pass        
            first_pose = self.get_ee_pose()
            print(first_pose)
            try:
              input("Please set the end effector to the second corner of your bounding rectangle, then click ENTER")
            except:
              pass        
            second_pose = self.get_ee_pose()
            print(second_pose)
            if first_pose.position.x > second_pose.position.x and first_pose.position.y > second_pose.position.y:
                lower_x = second_pose.position.x
                lower_y = second_pose.position.y
                
                upper_x = first_pose.position.x
                upper_y = first_pose.position.y

            elif first_pose.position.x < second_pose.position.x and first_pose.position.y < second_pose.position.y:
                lower_x = first_pose.position.x
                lower_y = first_pose.position.y
                
                upper_x = second_pose.position.x
                upper_y = second_pose.position.y

            else:
                print("Boundaries are incorrect, please try again")
                return False

            height = f_height

        self.lower_point = Point(lower_x, lower_y,height)
        self.lower_point = Pose(position=self.lower_point, orientation=upright)

        self.lower_opp_point = Point(lower_x, upper_y, height)
        self.lower_opp_point = Pose(position=self.lower_opp_point, orientation=upright)

        self.upper_point = Point(upper_x, upper_y, height)
        self.upper_point = Pose(position=self.upper_point, orientation=upright)

        self.upper_opp_point = Point(upper_x, lower_y, height)
        self.upper_opp_point = Pose(position=self.upper_opp_point, orientation=upright)

        return True


    def get_start_point(self, fixed=True):

        if fixed:
            self.start_point = Pose(position=Point(f_start_x, f_start_y, f_height), orientation=upright)
            return True
        else:
            try:
              input("Please set the end effector to the point you want it to always start at, then click ENTER")
            except:
              pass
            if not self.exceeds_bounds():
                self.start_point = self.get_ee_pose()
                print(self.start_point)
                return True
            else:
                print("Start state is incorrect, please try again")
                return False

    def go_to_start(self):
        print("Going to start")
        self.send_pose(self.start_point)
        

    def go_to(self, pose, high=False):
        pose = copy.deepcopy(pose)
        if high:
          pose.position.z = 1.05

        self.send_pose(pose)


    def fix_orientation(self):
        
        pose = self.get_ee_pose()
        pose.orientation = upright

        self.send_pose(pose)

    def orientation_violated(self):
        pose = self.get_ee_pose()

        quat = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]
        euler = euler_from_quaternion(quat)

        if euler[1] < 1.45 or euler[1] > 1.58:
            print(euler[1])
            return True
        else:
            return False


    def not_at_start(self):
        pose = self.get_ee_pose()

        for c,s in zip([pose.position.x, pose.position.y, pose.position.z],[self.start_point.position.x, self.start_point.position.y, self.start_point.position.z]):

            if c < s - 0.01 or c > s + 0.01:

                return True

        return False


    def exceeds_bounds(self):

        current_pose = self.get_ee_pose()

        if current_pose.position.x < self.lower_point.position.x or current_pose.position.y < self.lower_point.position.y or current_pose.position.x > self.upper_point.position.x or current_pose.position.y > self.upper_point.position.y:
            return True
        else:
            return False

if __name__ == "__main__":

    m = RLMoveIt()

    print(m.get_ee_pose())
    print(m.random_pose())
    print(m.exceeds_bounds())
