import numpy as np

class KalmanFilter:
    def __init__(self):
        # State vector: [x_center, y_center, width, height, vx, vy]
        # We track the center, size, and velocity of the bounding box.
        self.state = np.zeros(6)
        
        # State transition matrix (models the physics)
        # It predicts the next state based on the current one.
        self.F = np.array([[1, 0, 0, 0, 1, 0],  # x' = x + vx
                           [0, 1, 0, 0, 0, 1],  # y' = y + vy
                           [0, 0, 1, 0, 0, 0],  # w' = w
                           [0, 0, 0, 1, 0, 0],  # h' = h
                           [0, 0, 0, 0, 1, 0],  # vx' = vx
                           [0, 0, 0, 0, 0, 1]]) # vy' = vy

        # Measurement matrix (maps state to measurement space)
        # We only measure position and size, not velocity.
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0]])

        # Initial state covariance (our uncertainty about the initial state)
        self.P = np.eye(6) * 1000

        # Process noise covariance (uncertainty in the physics model)
        self.Q = np.eye(6) * 0.1

        # Measurement noise covariance (uncertainty from the YOLO detector)
        self.R = np.eye(4) * 10

    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        # Update the state with a new measurement
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain
        
        self.state = self.state + K @ y
        self.P = self.P - K @ self.H @ self.P