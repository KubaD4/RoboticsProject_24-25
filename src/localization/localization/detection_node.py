#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file detection_node.py
@brief Main ROS2 node for object detection and pose estimation.
@ingroup localization

This module implements a ROS2 node that uses the YOLO object detector and ICP-based
pose estimation to detect and localize blocks on a table. It exposes a service to provide
block information and publishes visualization data.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImg, PointCloud2
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time
from localization_interfaces.srv import BlockInfoAll
from . import constants
from . import pointcloud_utils
from . import pose_estimation

## @defgroup localization Localization Package
## @brief Contains modules related to object detection, point cloud processing, and pose estimation.
## @{
class DetectionNode(Node):
    """
    @brief ROS2 node for object detection and pose estimation.
    
    Processes incoming camera images and point cloud data to detect objects
    and estimate their 3D poses using YOLO and ICP.
    """
    
    def __init__(self):
        """
        @brief Constructor. Initializes storage, service, publishers/subscribers,
        and loads the YOLO model.
        """
        super().__init__('detection_node')
        
        # Initialize storage variables.
        self.current_detections = []
        self.detection_pointclouds = []
        self.latest_point_cloud = None
        self.latest_image = None
        self.stl_pointclouds = []
        self.processed_classes = set()
        self.object_poses = []

        # Service setup.
        self.srv = self.create_service(
            BlockInfoAll, 
            'localization_service', 
            self.handle_all_block_info_request
        )

        # Publishers and subscribers setup.
        self.setup_ros_communication()
        
        # Initialize YOLO model and cv_bridge.
        self.model = YOLO("/home/ubuntu/ros2_ws/src/localization/yolo_weights/best.pt")
        self.bridge = CvBridge()

        self.get_logger().info("Detection node initialized and waiting for service requests...")

    def setup_ros_communication(self):
        """
        @brief Sets up ROS publishers and subscribers.
        
        Creates a publisher for visualization images and subscribers
        for the raw camera image and point cloud topics.
        """
        self.vis_publisher = self.create_publisher(
            ROSImg, 
            '/object_pose_visualization',
            5
        )
        self.subscription_img = self.create_subscription(
            ROSImg, 
            '/camera/image_raw/image', 
            self.store_image, 
            5
        )
        self.subscription_pcl = self.create_subscription(
            PointCloud2, 
            '/camera/image_raw/points', 
            self.store_pointcloud, 
            5
        )

    def store_image(self, msg):
        """
        @brief Callback to store the latest camera image.
        @param msg The ROS image message.
        """
        self.latest_image = msg

    def store_pointcloud(self, msg):
        """
        @brief Callback to store the latest point cloud.
        @param msg The ROS PointCloud2 message.
        """
        self.latest_point_cloud = msg

    def handle_all_block_info_request(self, request, response):
        """
        @brief Service callback to handle block information requests.
        @param request The incoming service request.
        @param response The response to be populated with block info.
        @return The populated response message.
        
        Waits until both the latest image and point cloud data are available,
        then processes the data to detect objects, estimate their poses, and
        return block information.
        """
        try:
            self.get_logger().info("Received block info request - starting processing")
            
            # Wait until both image and point cloud messages are available.
            step = 0
            step_limit = 10  # seconds
            while step < step_limit:
                if self.latest_image is not None and self.latest_point_cloud is not None:
                    self.get_logger().warn(f"Image and point cloud ready after {step} seconds")
                    break
                time.sleep(1)
                step += 1
                self.get_logger().warn(f"Image or point cloud still not ready, waited {step} seconds")
            
            # If messages are still missing, return the response without further processing.
            if self.latest_image is None or self.latest_point_cloud is None:
                self.get_logger().error("No camera messages available")
                return response

            # Process the latest image.
            self.detect(self.latest_image)
            
            # Wait for processing to complete.
            max_attempts = 100  # about 10 seconds.
            attempt = 0
            
            while attempt < max_attempts:
                if self.object_poses:
                    # Build block info list.
                    block_infos = []
                    for pose in self.object_poses:
                        position = pose['position']
                        distance = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
                        
                        block_infos.append({
                            'distance': distance,
                            'name': pose['class_name'],
                            'position': position,
                            'quaternion': pose['quaternion']
                        })

                    # Sort by distance (descending).
                    block_infos.sort(key=lambda x: x['distance'], reverse=False)

                    # Fill response fields.
                    response.block_names = [block['name'] for block in block_infos]
                    response.positions_x = [float(block['position'][0]) for block in block_infos]
                    response.positions_y = [float(block['position'][1]) for block in block_infos]
                    response.positions_z = [float(block['position'][2]) for block in block_infos]
                    response.orientations_x = [float(block['quaternion'][0]) for block in block_infos]
                    response.orientations_y = [float(block['quaternion'][1]) for block in block_infos]
                    response.orientations_z = [float(block['quaternion'][2]) for block in block_infos]
                    response.orientations_w = [float(block['quaternion'][3]) for block in block_infos]
                    
                    self.get_logger().info(f"Successfully processed {len(block_infos)} blocks")
                    
                    # Clear data for the next request.
                    self.object_poses = []
                    self.current_detections = []
                    self.detection_pointclouds = []
                    self.get_logger().info("Detection node waiting for service requests...")
                    return response
                
                attempt += 1
                time.sleep(0.1)
            
            # Processing timed out.
            self.get_logger().warn("Processing timed out - no blocks detected")
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error processing request: {str(e)}")
            return response

    def detect(self, ros2_img):
        """
        @brief Processes an incoming image for object detection using YOLO.
        @param ros2_img The ROS image message.
        
        Converts the ROS image to an OpenCV image, performs detection,
        loads the corresponding STL models if necessary, and extracts
        portions of the point cloud.
        """
        try:
            self.get_logger().warn("DETECT")
            cv_image = self.bridge.imgmsg_to_cv2(ros2_img, desired_encoding='passthrough')
            results = self.model.predict(source=cv_image, conf=0.25)
            
            self.current_detections = []
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = constants.CLASS_NAMES[cls]

                    # Load STL model if not already loaded.
                    self.load_stl_model(cls)
                    
                    self.current_detections.append({
                        'bbox': xyxy,
                        'confidence': conf,
                        'class': cls,
                        'class_name': class_name
                    })
                    
                if self.latest_point_cloud is not None:
                    self.extract_pointcloud_portions()
                    
        except Exception as e:
            self.get_logger().error(f"Error in detection: {e}")

    def load_stl_model(self, class_idx):
        """
        @brief Loads and converts an STL model to a point cloud for a given class.
        @param class_idx The class index of the detected object.
        @return True if the model was loaded successfully, False otherwise.
        """
        if class_idx in self.processed_classes or class_idx >= len(constants.CLASS_NAMES):
            return False
                
        class_name = constants.CLASS_NAMES[class_idx]
        if class_name == 'undefined':
            return False
                
        stl_path = os.path.join(
            constants.BASE_STL_PATH, 
            class_name, 
            'mesh', 
            f'{class_name}.stl'
        )
        
        points = pointcloud_utils.load_stl_model(stl_path)
        if points is not None:
            self.stl_pointclouds.append({
                'label': class_name,
                'points': points,
                'class_idx': class_idx
            })
            self.processed_classes.add(class_idx)
            return True
        
        return False

    def extract_pointcloud_portions(self):
        """
        @brief Extracts point cloud segments corresponding to each detected object.
        
        For each detection, extracts a portion of the latest point cloud using the bounding box,
        preprocesses the points, and then processes them via ICP to estimate object poses.
        """
        if not self.latest_point_cloud or not self.current_detections:
            return
            
        self.detection_pointclouds = []
        
        for detection in self.current_detections:
            points = pointcloud_utils.extract_points_from_ros_msg(
                self.latest_point_cloud,
                detection['bbox']
            )
            
            if len(points) > 0:
                processed_points = pointcloud_utils.preprocess_pointcloud(points)
                
                self.detection_pointclouds.append({
                    'points': processed_points,
                    'class_idx': detection['class'],
                    'bbox': detection['bbox']
                })

        # Run ICP alignment if there are valid point clouds.
        if self.detection_pointclouds:
            self.process_poses()

    def process_poses(self):
        """
        @brief Processes the detected objects to estimate their poses using ICP.
        
        For each detected object, retrieves the corresponding STL model and runs the ICP algorithm.
        If successful, extracts pose parameters and logs the estimated pose.
        """
        self.object_poses = []
        
        for detected in self.detection_pointclouds:
            stl_model = next(
                (model for model in self.stl_pointclouds 
                 if model['class_idx'] == detected['class_idx']), 
                None
            )
            
            if stl_model is None:
                self.get_logger().warn(f"No STL model found for class {detected['class_idx']}")
                continue
                
            R, t, error, converged = pose_estimation.icp_align(
                stl_model['points'], 
                detected['points'],
                constants.MAX_ICP_ITERATIONS,
                constants.ICP_TOLERANCE
            )
            
            if not converged or R is None or t is None:
                self.get_logger().warn(f"ICP failed for object {constants.CLASS_NAMES[detected['class_idx']]}")
                continue
                
            pose_params = pose_estimation.extract_pose_parameters(R, t)
            
            self.object_poses.append({
                'class_idx': detected['class_idx'],
                'class_name': constants.CLASS_NAMES[detected['class_idx']],
                'position': pose_params['position'],
                'euler_angles': pose_params['euler_angles'],
                'quaternion': pose_params['quaternion'],
                'alignment_error': error,
                'bbox': detected['bbox']
            })
            
            self.get_logger().info(
                f"\nPose estimated for {constants.CLASS_NAMES[detected['class_idx']]}:\n"
                f"Position (x,y,z): {pose_params['position']}\n"
                f"Orientation (r,p,y): {pose_params['euler_angles']}\n"
                f"Alignment error: {error}"
            )
## @}
            
def main(args=None):
    """
    @brief Main entry point for the detection node.
    
    Initializes the ROS2 node, spins it until shutdown, and cleans up resources.
    """
    rclpy.init(args=args)
    detector = DetectionNode()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("Shutting down DetectionNode...")
    finally:
        cv2.destroyAllWindows()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
