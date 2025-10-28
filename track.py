import numpy as np
from kalman import KalmanFilter

class Track:
    def __init__(self, track_id, initial_bbox):
        self.track_id = track_id
        self.kf = KalmanFilter()
        
        # Initialize the filter with the first detection
        self.kf.state[:4] = self.bbox_to_state(initial_bbox)
        
        self.bbox = initial_bbox
        self.time_since_update = 0 # Frames since last detection
        self.hits = 1 # Number of times this track has been detected

    def bbox_to_state(self, bbox):
        # Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        return np.array([center_x, center_y, width, height])

    def predict(self):
        # Predict the next position using the Kalman Filter
        predicted_state = self.kf.predict()
        
        # Convert predicted state back to a bounding box for matching
        cx, cy, w, h, _, _ = predicted_state
        self.bbox = [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
        self.time_since_update += 1

    def update(self, bbox):
        # Update the Kalman Filter with the new detection
        measurement = self.bbox_to_state(bbox)
        self.kf.update(measurement)
        
        # Update track properties
        self.bbox = bbox
        self.time_since_update = 0
        self.hits += 1