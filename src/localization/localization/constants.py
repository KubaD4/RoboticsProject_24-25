#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file constants.py
@brief Constants and configurations used across the detection system.
@ingroup localization

This module contains definitions for file paths, class names,
and parameters used by the detection and pose estimation algorithms.
"""
import os

## Base path for STL models.
BASE_STL_PATH = "/home/ubuntu/ros2_ws/src/Models-2"

## List of class names for object detection.
CLASS_NAMES = [
    'X1-Y1-Z2', 'X1-Y2-Z1', 'X1-Y2-Z2', 'X1-Y2-Z2-CHAMFER',
    'X1-Y2-Z2-TWINFILLET', 'X1-Y3-Z2', 'X1-Y3-Z2-FILLET',
    'X1-Y4-Z1', 'X1-Y4-Z2', 'X2-Y2-Z2', 'X2-Y2-Z2-FILLET',
    'undefined'
]

## Maximum number of ICP iterations.
MAX_ICP_ITERATIONS = 300

## Tolerance for ICP convergence.
ICP_TOLERANCE = 1e-7
