import numpy as np
import pandas as pd
import matplotlib
from pykalman import AdditiveUnscentedKalmanFilter

class UKFFilter:
    def __init__(self, data, initial_state, dt=1/60):
        # raw data
        self.data = data
        
        # l
        self.num_stamps_length = data.shape[0]
        # s_0
        self.initial_state = initial_state

        # Bone point pair information
        self.bone_pairs = [
            ('head', 'chin'),
            ('chin', 'Neck'),
            ('Neck', 'RShoulder'),
            ('Neck', 'LShoulder'),
            ('RShoulder', 'RElbow'),
            ('RElbow', 'RWrist'),
            ('RWrist', 'Rhand'),
            ('LShoulder', 'LElbow'),
            ('LElbow', 'LWrist'),
            ('LWrist', 'Lhand'),
            ('RShoulder', 'RHip'),
            ('LShoulder', 'LHip'),
            ('RHip', 'LHip'),
            ('RHip', 'RKnee'),
            ('RKnee', 'RAnkle'),
            ('RAnkle', 'RHeel'),
            ('RHeel', 'RBigToe'),
            ('LHip', 'LKnee'),
            ('LKnee', 'LAnkle'),
            ('LAnkle', 'LHeel'),
            ('LHeel', 'LBigToe')
        ]
        self.dim = initial_state.shape[0]
        self.dt = dt
        self.dim_x = self.dim * 2  # Dimensions of the state, including position and velocity
        self.dim_z = self.dim  # Dimensions of the observation

        # Initialization process noise and observation noise
        self.process_noise = np.eye(self.dim_x) * 0.001   # Adjust according to the dataset
        self.measurement_noise = np.eye(self.dim_z) * 0.1 # Adjust according to the dataset

        # Initialize the state and covariance matrix
        self.x = np.hstack((initial_state, np.zeros(self.dim)))  # Includes position and velocity
        self.P = np.eye(self.dim_x) * 1e-9                # Adjust according to the dataset

        self.ukf = AdditiveUnscentedKalmanFilter(
            transition_functions=self.fx,
            observation_functions=self.hx,
            transition_covariance=self.process_noise,
            observation_covariance=self.measurement_noise,
            initial_state_mean=self.x,
            initial_state_covariance=self.P
        )
        # fine-grained filtering: distances calculating
        self.desired_distances = self.calculate_desired_distances()

    def calculate_desired_distances(self):
        all_distances = {pair: [] for pair in self.bone_pairs}
        for t in range(self.num_stamps_length):
            for (joint1, joint2) in self.bone_pairs:

                # Gets the bone point index
                idx1 = self.get_joint_index(joint1)
                idx2 = self.get_joint_index(joint2)

                # Get the bone point coordinates
                point1 = self.data[t, idx1 * 3:(idx1 + 1) * 3]
                point2 = self.data[t, idx2 * 3:(idx2 + 1) * 3]
                distance = np.linalg.norm(point1 - point2)
                all_distances[(joint1, joint2)].append(distance)

        mean_distances = {}
        for pair, distances in all_distances.items():
            # Sort all distances
            sorted_distances = np.sort(distances)

            # Take the middle 50% of the data
            middle_index = len(sorted_distances) // 4
            middle_distances = sorted_distances[middle_index: -middle_index]

            # Calculate the mean of the middle 50% of the data
            mean_distances[pair] = np.mean(middle_distances)

        return mean_distances

    def apply_finegrained_filtering(self, error_tolerance=0):
        for (joint1, joint2) in self.bone_pairs:
            idx1 = self.get_joint_index(joint1)
            idx2 = self.get_joint_index(joint2)
            point1 = self.x[idx1*3:(idx1+1)*3]
            point2 = self.x[idx2*3:(idx2+1)*3]
            distance = np.linalg.norm(point1 - point2)

            if distance != 0:
                desired_distance = self.desired_distances[(joint1, joint2)]
                min_distance = desired_distance - error_tolerance
                max_distance = desired_distance + error_tolerance

                if min_distance <= distance <= max_distance:
                    continue

                midpoint = (point1 + point2) / 2
                direction = (point2 - point1) / distance
                new_point1 = midpoint - direction * (desired_distance / 2)
                new_point2 = midpoint + direction * (desired_distance / 2)
                self.x[idx1 * 3:(idx1 + 1) * 3] = new_point1
                self.x[idx2 * 3:(idx2 + 1) * 3] = new_point2

    def get_joint_index(self, joint_name):
        joint_index_map = {
            'head': 0, 'chin': 1, 'Neck': 2, 'RShoulder': 3, 'RElbow': 4, 'RWrist': 5, 'Rhand': 6, 'LShoulder': 7, 'LElbow': 8, 'LWrist': 9,
            'Lhand': 10, 'RHip': 11, 'RKnee': 12, 'RAnkle': 13, 'RHeel': 14, 'RBigToe': 15, 'LHip': 16, 'LKnee': 17, 'LAnkle': 18, 'LHeel': 19,
            'LBigToe': 20
        }
        return joint_index_map[joint_name]

    def fx(self, x):
        # Consider velocity
        position = x[:self.dim]
        velocity = x[self.dim:]

        # dt = 1/60
        new_position = position + velocity * self.dt

        return np.hstack((new_position, velocity))

    def hx(self, x):

        position = x[:self.dim]
        observed_value = position

        return observed_value

    def filter(self):

        filtered_data = np.zeros((self.num_stamps_length, self.dim))
        for i in range(self.num_stamps_length):
            z = self.data[i]  # Gets the current observation
            self.x, self.P = self.ukf.filter_update(self.x, self.P, z)
            filtered_data[i] = self.x[:self.dim]
            self.apply_finegrained_filtering()  # Apply fine-grained filtering
        return filtered_data