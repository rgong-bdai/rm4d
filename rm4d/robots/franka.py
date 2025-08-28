import numpy as np
import pybullet as p
from .base import RobotBase
from .assets import FRANKA_URDF


class Franka(RobotBase):
    def __init__(self, simulator, base_pos=None, base_orn=None, urdf_fn=None):
        super().__init__(simulator, base_pos, base_orn)
        if urdf_fn is None:
            urdf_fn = FRANKA_URDF
        self._robot_id = self.sim.bullet_client.loadURDF(urdf_fn,
                                                         self.base_pos, self.base_orn, useFixedBase=True)

        # robot properties
        # self._end_effector_link_id = 10
        self._end_effector_link_id = 11
        self._arm_joint_ids = [0, 1, 2, 3, 4, 5, 6]
        self.home_conf = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807, -2.342050140544315,
                          0.029840531355804868, 1.5411935298621688, 0.7534486589746342]

        # determine joint limits
        limits = []
        for i in self.arm_joint_ids:
            limits.append(self.sim.bullet_client.getJointInfo(self.robot_id, i)[8:10])
        self._joint_limits = np.asarray(limits)

        # set initial robot configuration
        self.reset_joint_pos(self.home_conf)
        for finger_id in [9, 10]:
            self.sim.bullet_client.resetJointState(self.robot_id, finger_id, 0.04)

    @property
    def range_radius(self) -> float:
        return 1.05

    @property
    def range_z(self) -> float:
        return 1.35

    @property
    def end_effector_link_id(self):
        return self._end_effector_link_id

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def arm_joint_ids(self):
        return self._arm_joint_ids

    @property
    def joint_limits(self):
        return self._joint_limits

    def in_self_collision(self):
        # check self-collision
        ignore_links = [7, 11]  # they do not have a collision shape
        first_links = [0, 1, 2, 3, 4, 5]  # 6 cannot collide with the fingers due to kinematics

        for first_link in first_links:
            # skip links that are next to each other (supposed to be in contact) plus all the ignore links
            check_links = [link for link in np.arange(first_link + 2, self.end_effector_link_id + 1) if
                           link not in ignore_links]
            for check_link in check_links:
                collision = self.sim.links_in_collision(self.robot_id, first_link, self.robot_id, check_link)
                if collision:
                    return True
        return False

    def get_jacobian(self):
        """
        Computes the Jacobian matrix for the end effector link based on the robot's current joint config.
        :return: (6, 7) ndarray, Jacobian matrix.
        """
        # pybullet also needs the finger joints for calculating the Jacobian, so actually gives a (3, 9) matrix.
        # however, the finger joints do not affect end-effector position, so we do not consider them and only provide
        # the Jacobian for the arm joints, as a (3, 7) matrix.
        zero_vec = [0.0] * (len(self.arm_joint_ids) + 2)
        local_pos = [0.0, 0.0, 0.0]
        joint_pos = list(self.joint_pos()) + [0.04, 0.04]
        jac_t, jac_r = p.calculateJacobian(self.robot_id, self.end_effector_link_id, local_pos, joint_pos,
                                           zero_vec, zero_vec)
        jac_t = np.asarray(jac_t)
        jac_r = np.asarray(jac_r)
        jac = np.concatenate([jac_t, jac_r], axis=0)
        return jac[:, :7]  # remove fingers

    def _do_inverse_kinematics(self, pos, quat, start_q, rest_q, n_iterations, threshold):
        self.reset_joint_pos(start_q)

        # get joint ranges and limits
        lower_limits = self.joint_limits[:, 0]
        upper_limits = self.joint_limits[:, 1]
        joint_ranges = upper_limits - lower_limits

        # include finger joints
        lower_limits = lower_limits.tolist() + [0.0, 0.0]
        upper_limits = upper_limits.tolist() + [0.04, 0.04]
        joint_ranges = joint_ranges.tolist() + [0.04, 0.04]
        rest_poses = rest_q.tolist() + [0.04, 0.04]

        # execute IK
        full_joint_positions = self.sim.bullet_client.calculateInverseKinematics(
            self.robot_id, self.end_effector_link_id, pos, quat,
            lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
            maxNumIterations=n_iterations, residualThreshold=threshold)

        return np.array(full_joint_positions)[self.arm_joint_ids]
