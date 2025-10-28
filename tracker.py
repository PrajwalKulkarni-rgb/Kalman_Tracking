import numpy as np
from scipy.optimize import linear_sum_assignment
from track import Track

def iou(bbox1, bbox2):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):
        # 1. Predict the next state for all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Match detections with tracks using the Hungarian algorithm
        if len(self.tracks) > 0 and len(detections) > 0:
            # Create a cost matrix based on IoU
            cost_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = 1 - iou(track.bbox, det) # We use 1 - IoU as a cost
            
            # Solve the assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 3. Update matched tracks, handle unmatched tracks/detections
            matched_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < 0.7: # IoU threshold of 0.3
                    self.tracks[r].update(detections[c])
                    matched_indices.add(c)
            
            unmatched_detections = [det for i, det in enumerate(detections) if i not in matched_indices]
        else:
            unmatched_detections = detections
            
        # 4. Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_track = Track(self.next_track_id, det)
            self.tracks.append(new_track)
            self.next_track_id += 1
            
        # 5. Remove old tracks that haven't been seen for a while
        self.tracks = [t for t in self.tracks if t.time_since_update <= 5]
        
        return self.tracks