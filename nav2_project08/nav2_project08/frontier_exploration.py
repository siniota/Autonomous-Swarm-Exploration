#!/usr/bin/env python3
"""
DEPRECATED — This single-robot explorer is superseded by
frontier_coordinator.py, which handles both robots using the merged map.
Kept for reference only; do NOT launch alongside frontier_coordinator.
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Change this variable to target 'robot1' or 'robot2'
        self.robot_name = 'robot1'
        
        # Subscribers and Publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',  # Subscribes to the global shared SLAM map
            self.map_callback,
            10)
            
        self.goal_pub = self.create_publisher(
            PoseStamped,
            f'/{self.robot_name}/goal_pose',
            10)
            
        self.get_logger().info(f"Frontier Explorer initialized for {self.robot_name}")

    def map_callback(self, msg: OccupancyGrid):
        # 1. Convert flat OccupancyGrid array into a 2D numpy grid
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        
        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        
        # 2. Extract unknown space (-1) and free space (0) boundaries
        # Create binary masks for edge detection
        unknown_mask = np.where(grid == -1, 255, 0).astype(np.uint8)
        free_mask = np.where(grid == 0, 255, 0).astype(np.uint8)
        
        # Detect edges of free space to locate exploration frontiers
        edges = cv2.Canny(free_mask, 100, 200)
        
        # Frontiers are free cells adjacent to unknown cells
        frontier_mask = cv2.bitwise_and(edges, unknown_mask)
        
        # 3. Cluster frontier pixels into groups using connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask)
        
        best_frontier_map_x = None
        best_frontier_map_y = None
        max_size = 0
        
        # Loop through found clusters (skip label 0, which is the background)
        for i in range(1, num_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out small clusters / noise
            if size > 5 and size > max_size:
                max_size = size
                # Get the centroid of the pixel cluster
                cx, cy = centroids[i]
                
                # 4. Transform pixel coordinates back into real-world Map coordinates
                best_frontier_map_x = origin_x + (cx * resolution)
                best_frontier_map_y = origin_y + (cy * resolution)
                
        # 5. Publish the tracking goal if a valid frontier cluster was found
        if best_frontier_map_x is not None:
            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'
            
            goal.pose.position.x = best_frontier_map_x
            goal.pose.position.y = best_frontier_map_y
            goal.pose.orientation.w = 1.0  # Default facing direction
            
            self.goal_pub.publish(goal)
            self.get_logger().info(f"Dispatched tracking goal: X={best_frontier_map_x:.2f}, Y={best_frontier_map_y:.2f}")
        else:
            self.get_logger().warn("Exploration complete! No new frontiers found.")

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

