#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file pose_estimation.py
@brief Functions for pose estimation using the ICP algorithm.
@ingroup localization

Provides functions to compute an initial alignment, find closest points,
calculate rigid transformations, apply transformations, compute errors,
and perform ICP alignment. Also extracts pose parameters (position,
Euler angles, quaternion) from the computed transformation.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import time
import copy
from scipy.spatial import KDTree

def estimate_initial_pose(source_points, target_points):
    """
    @brief Estimates an initial rigid transformation using principal component analysis.
    @param source_points Array of source 3D points.
    @param target_points Array of target 3D points.
    @return A tuple (R, t) where R is the rotation matrix and t is the translation vector.
    
    Computes the centroids and covariance matrices of both point sets, then uses the eigenvectors
    of the covariance matrices to compute an initial rotation. Adjusts the rotation if necessary
    to ensure a right-handed coordinate system.
    """
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    source_covariance = np.cov(source_centered.T)
    target_covariance = np.cov(target_centered.T)
    
    source_eigvals, source_eigvecs = np.linalg.eigh(source_covariance)
    target_eigvals, target_eigvecs = np.linalg.eigh(target_covariance)
    
    R = target_eigvecs @ source_eigvecs.T
    
    if np.linalg.det(R) < 0:
        source_eigvecs[:, -1] *= -1
        R = target_eigvecs @ source_eigvecs.T
    
    t = target_centroid - (R @ source_centroid)
    
    return R, t

def find_closest_points(source_points, target_points):
    """
    @brief Finds corresponding points between two point sets using a KD-tree.
    @param source_points Array of source 3D points.
    @param target_points Array of target 3D points.
    @return A tuple (matched_target, matched_source) containing the corresponding points.
    
    Uses a KD-tree to query the nearest neighbor for each source point.
    Only matches with distances below the 90th percentile are considered valid.
    """
    tree = KDTree(target_points)
    distances, indices = tree.query(source_points)
    good_matches_mask = distances < np.percentile(distances, 90)
    return target_points[indices[good_matches_mask]], source_points[good_matches_mask]

def calculate_transformation(source_points, target_points):
    """
    @brief Calculates the optimal rigid transformation between two point sets.
    @param source_points Array of source 3D points.
    @param target_points Array of target 3D points.
    @return A tuple (R, t) representing the rotation matrix and translation vector.
    
    Computes the centroids, centers the points, and calculates the cross-covariance matrix.
    Then uses Singular Value Decomposition (SVD) to compute the optimal rotation and translation.
    """
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = target_centroid - (R @ source_centroid)
    
    return R, t

def apply_transformation(points, R, t):
    """
    @brief Applies a rigid transformation to a set of points.
    @param points Array of 3D points.
    @param R Rotation matrix.
    @param t Translation vector.
    @return Transformed array of 3D points.
    """
    return (R @ points.T).T + t

def calculate_error(source_points, target_points):
    """
    @brief Calculates the root-mean-square error (RMSE) between two point sets.
    @param source_points Array of source 3D points.
    @param target_points Array of target 3D points.
    @return The RMSE as a float.
    """
    distances = np.linalg.norm(source_points - target_points, axis=1)
    return np.sqrt(np.mean(distances**2))

def icp_align(source_points, target_points, max_iterations, tolerance):
    """
    @brief Performs Iterative Closest Point (ICP) alignment.
    @param source_points Array of source 3D points.
    @param target_points Array of target 3D points.
    @param max_iterations Maximum number of ICP iterations.
    @param tolerance Convergence tolerance for the change in error.
    @return A tuple (R, t, error, converged) where R is the rotation matrix, t is the translation,
            error is the final RMSE, and converged is a boolean indicating if ICP converged.
    
    The function starts with an initial pose estimation and iteratively refines the transformation
    until convergence or until the maximum number of iterations is reached.
    """
    if len(source_points) < 3 or len(target_points) < 3:
        return None, None, float('inf'), False
        
    try:
        start_time = time.time()
        R_current, t_current = estimate_initial_pose(source_points, target_points)
        transformed_source = copy.deepcopy(source_points)
        prev_error = float('inf')
        
        for iteration in range(max_iterations):
            transformed_source = apply_transformation(source_points, R_current, t_current)
            matched_target, matched_source = find_closest_points(transformed_source, target_points)
            
            if len(matched_source) < 3:
                return None, None, float('inf'), False
            
            R_step, t_step = calculate_transformation(matched_source, matched_target)
            R_current = R_step @ R_current
            t_current = R_step @ t_current + t_step
            
            current_error = calculate_error(matched_source, matched_target)
            
            if abs(prev_error - current_error) < tolerance:
                return R_current, t_current, current_error, True
                
            prev_error = current_error
            
        return R_current, t_current, prev_error, False
        
    except Exception as e:
        print(f"Error in ICP alignment: {str(e)}")
        return None, None, float('inf'), False

def extract_pose_parameters(R, t):
    """
    @brief Extracts pose parameters from a rotation matrix and translation vector.
    @param R Rotation matrix.
    @param t Translation vector.
    @return A dictionary containing position, Euler angles (in degrees), and quaternion.
    
    Uses the scipy Rotation module to convert the rotation matrix to Euler angles and quaternion.
    """
    position = t.flatten()
    rotation = Rotation.from_matrix(R)
    euler_angles = rotation.as_euler('xyz', degrees=True)
    quaternion = rotation.as_quat()
    
    return {
        'position': position,
        'euler_angles': euler_angles,
        'quaternion': quaternion
    }
