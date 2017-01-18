import os
import re

import numpy as np
from scipy import misc
import navpy

import random
import cv2

def convert_gps(coords, ref_pt):
    '''
    Convert a collection of GPS coordinates (LLA) into local NED
    coordinates centered at ref_pt.

    Assumes that lat and lon are given in degrees and alt is given in
    meters.
    '''
    lat = coords[:, 0]
    lon = coords[:, 1]
    alt = coords[:, 2]
    
    local_ned = navpy.lla2ned(lat, lon, alt, ref_pt[0], ref_pt[1], ref_pt[2])

def normalize_gps(coords, ref_pt):
    '''
    Convert and normalize a collection of GPS coordinates so that the
    (x,y,z) coordinates in the NED frame are normalized to the range
    [-1,1].
    '''
    converted = convert_gps(coords, ref_pt)
    means = np.mean(converted, axis=0)
    stds = np.std(converted, axis=0)

    return (x - means)/stds

class RenoDriving(object):
    
    def __init__(self, image_path, coord_path):
        self.images = {}
        self.coords = {}
        self.NUM_COORDS = 3

        self.create_dataset(image_path, coord_path)
        
        if set(self.images.keys()) != set(self.coords.keys()):
            print("Keys and coordinates don't match!!")

        self.processed = []
        
        self.epochs = 0

    def create_dataset(self, image_path, coord_path):
        # get images
        for subdir, dirs, files in os.walk(image_path):
            idx = 0
            for f in files:
                match = re.search("\d*", f)
                num = int(match.group(0))
                img = cv2.imread(os.path.join(subdir,f))
                self.images[num] = 1.0 - img #cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                idx += 1

        # get coords
        for subdir, dirs, files in os.walk(coord_path):
            for f in files:
                match = re.search("\d*", f)
                num = int(match.group(0))
                
                f = open(os.path.join(subdir, f), 'r')
                for line in f:
                    parts = line.split()
                    if len(parts) != self.NUM_COORDS:
                        break
                    self.coords[num] = [float(x) for x in parts]

    def next_batch(self, batch_size):
        # get indices
        keys = self.images.keys()
        candidates = list(set(keys) - set(self.processed))
        
        rows = self.images[candidates[0]].shape[0]
        cols = self.images[candidates[0]].shape[1]
        chans = self.images[candidates[0]].shape[2]

        # handle case where we have used all the indices but have more
        # to do
        if len(candidates) < batch_size:
            # first add all the candidates
            indices = candidates

            # increment epochs
            self.epochs += 1

            # compute the number remaining to add
            remainder = batch_size - len(candidates)
            remainder_indices = random.sample(keys, remainder)
            indices += remainder_indices

            # reset processed
            self.processed = remainder_indices

        else:
            indices = random.sample(candidates, batch_size)
            self.processed += indices

        # pack images into array
        image_array = np.zeros((batch_size, rows, cols, chans))
        idx = 0
        for i in indices:
            image_array[idx] = cv2.cvtColor(self.images[i], cv2.COLOR_BGR2RGB)
            idx += 1

        # pack coords into array
        coord_array = np.zeros((batch_size, self.NUM_COORDS))
        idx = 0
        for i in indices:
            coord_array[idx] = self.coords[i]
            idx += 1

        return (image_array, coord_array)
