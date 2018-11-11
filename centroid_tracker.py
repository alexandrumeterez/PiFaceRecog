from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

MAX_DISSAPEARED_FRAMES = 50
class CentroidTracker():
    def __init__(self):
        self.next_object_id = 0
        self.dissapeared = OrderedDict() #keeps track for each object how many frames it was out of focus
        self.objects = OrderedDict() #keeps track for each object of its centroid
    
    #add new object to tracked objects
    def register(self, centroid):
        self.objects[self.next_object_id] = centroid #add object to dict
        self.dissapeared[self.next_object_id] = 0 #initialize its number of dissapeared frames to 0
        self.next_object_id += 1 #increment next object id for the next object
    
    #used to remove objects that were out of focus for more than MAX_DISSAPEARED_FRAMES frames
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.dissapeared[object_id]
        
    def update(self, rects):
        if len(rects) == 0:
            #if there are no new objects to be tracked
            for obj_id in self.dissapeared.keys():
                #increment all tracked objects number of dissapeared frames with 1
                #and delete those that are above the threshold
                self.dissapeared[obj_id] += 1
                if self.dissapeared[obj_id] >= MAX_DISSAPEARED_FRAMES:
                    self.deregister(obj_id)
            return self.objects
        else:
            #initialize all input centroids with zero
            input_centroids = np.zeros((len(rects), 2), dtype="int")
            #otherwise, if we have detected objects
            #check if we are tracking any
            #if not, register new objects
            for (i, (startX, startY, endX, endY)) in enumerate(rects):
                #calculate all input centroids
            
                input_centroids[i] = (int((startX+endX)/2), int((startY+endY)/2))
            if len(self.objects) == 0:
                for i in range(0, len(input_centroids)):
                    self.register(input_centroids[i])
            #otherwise, we need to update the tracked objects
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())
                
                D = dist.cdist(np.array(object_centroids), input_centroids)
                #find the closest input centroid for each of the tracked centroids
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                
                #determine if we need to update, register or deregister
                used_rows = set()
                used_cols = set()
                
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols:
                        continue
                    obj_id = object_ids[row]
                    self.objects[obj_id] = input_centroids[col]
                    self.dissapeared[obj_id] = 0
                    used_rows.add(row)
                    used_cols.add(col)
                unused_rows = set(range(0, D.shape[0])).difference(used_rows)
                unused_cols = set(range(0, D.shape[1])).difference(used_cols)
                if D.shape[0] >= D.shape[1]:
                    for row in unused_rows:
                        obj_id = object_ids[row]
                        self.dissapeared[obj_id] += 1
                        
                        if self.dissapeared[obj_id] > MAX_DISSAPEARED_FRAMES:
                            self.deregister(obj_id)
                else:
                    for col in unused_cols:
                        self.register(input_centroids[col])
        return self.objects