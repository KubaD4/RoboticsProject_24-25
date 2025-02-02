#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file pointcloud_utils.py
@brief Utility functions for point cloud processing and manipulation.
@ingroup localization

Provides functions for preprocessing point clouds, loading STL models,
and extracting point cloud segments from ROS PointCloud2 messages.
"""

import numpy as np
import trimesh
import struct
from scipy.spatial import KDTree
import os
from . import constants

def preprocess_pointcloud(points, voxel_size=0.01):
    """
    @brief Preprocesses a point cloud by removing outliers and the ground plane.
    @param points List or array of 3D points.
    @param voxel_size Voxel size for downsampling (unused in this implementation).
    @return A numpy array of filtered points.
    
    This function removes statistical outliers and then removes points that
    lie on the ground plane (using the 10th percentile of the z-values).
    """
    try:
        points_array = np.array(points)
        
        if len(points_array) == 0:
            return np.array([])
        if len(points_array.shape) != 2 or points_array.shape[1] != 3:
            return np.array([])
        
        # Statistical outlier removal.
        mean = np.mean(points_array, axis=0)
        std = np.std(points_array, axis=0)
        inliers_mask = np.all(np.abs(points_array - mean) <= 1.5 * std, axis=1)
        filtered_points = points_array[inliers_mask]
        
        if len(filtered_points) == 0:
            return np.array([])
        
        # Remove ground plane points.
        z_threshold = np.percentile(filtered_points[:, 2], 10)
        above_ground_mask = filtered_points[:, 2] > z_threshold
        filtered_points = filtered_points[above_ground_mask]
        
        return filtered_points
        
    except Exception as e:
        print(f"Error in pointcloud preprocessing: {str(e)}")
        return np.array([])

def load_stl_model(stl_path):
    """
    @brief Loads an STL model and converts it into a point cloud.
    @param stl_path Path to the STL file.
    @return A numpy array of sampled points from the mesh, or None if loading fails.
    """
    try:
        if not os.path.exists(stl_path):
            return None
            
        mesh = trimesh.load(stl_path)
        points = mesh.sample(5000)
        
        return points if len(points) > 0 else None
        
    except Exception as e:
        print(f"Error loading STL model: {str(e)}")
        return None

def extract_points_from_ros_msg(point_cloud_msg, bbox, margin=4):
    """
    @brief Extracts a subset of points from a ROS PointCloud2 message.
    @param point_cloud_msg The ROS PointCloud2 message.
    @param bbox Bounding box (x1, y1, x2, y2) defining the region of interest.
    @param margin Pixel margin to extend the bounding box.
    @return A numpy array of extracted 3D points.
    
    Iterates through the region defined by the bounding box (with added margin)
    and extracts 3D points if they contain valid (finite) coordinates.
    """
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2 = min(point_cloud_msg.width, x2 + margin)
    y2 = min(point_cloud_msg.height, y2 + margin)
    
    points = []
    for row in range(y1, y2):
        for col in range(x1, x2):
            offset = row * point_cloud_msg.row_step + col * point_cloud_msg.point_step
            (x, y, z) = struct.unpack_from('fff', point_cloud_msg.data, offset)
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z])
                
    return np.array(points)
