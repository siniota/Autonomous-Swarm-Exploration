#!/usr/bin/env python3
"""
Multi-Robot Frontier Coordinator

Fixes applied from review:
  1. Detects frontiers on the MERGED /map (not per-robot maps)
  2. Uses morphological dilation instead of fragile Canny edge detection
  3. Deduplicates nearby frontier clusters
  4. Assigns frontiers by greedy nearest-distance (not list index)
  5. Uses a timer for replanning instead of spamming on every map callback
  6. Publishes goals in the 'map' frame (the merged frame)
  7. Uses TF2 to get each robot's current position for distance calculation
"""
import math

import cv2  # type: ignore[import-untyped]
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                        QoSReliabilityPolicy)
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class MultiRobotFrontierCoordinator(Node):
    def __init__(self):
        super().__init__('frontier_coordinator')

        self.robots = ['robot1', 'robot2']
        self.assigned_goals = {robot: None for robot in self.robots}

        # TF2 for looking up each robot's current position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS matching map_merge_node's publisher (RELIABLE + TRANSIENT_LOCAL)
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.history = QoSHistoryPolicy.KEEP_LAST

        # Subscribe to the MERGED map (single source of truth)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, map_qos)

        self.latest_map = None

        # Goal publishers per robot
        self.pubs = {}
        for name in self.robots:
            self.pubs[name] = self.create_publisher(
                PoseStamped, f'/{name}/goal_pose', 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/frontiers_markers', 10)

        # Timer-based replanning instead of goal-spamming on every map update
        self.replan_period = 3.0  # seconds
        self.replan_timer = self.create_timer(self.replan_period, self.replan)

        # Tuning knobs
        self.dedup_radius = 0.5           # metres – merge centroids closer than this
        self.min_frontier_size = 5        # pixels – ignore tiny noise clusters
        self.goal_reached_threshold = 0.5 # metres – consider goal reached within this
        self.min_goal_distance = 0.8      # metres – skip frontiers closer than this

        self.get_logger().info(
            "Frontier Coordinator started (merged-map mode, "
            f"replan every {self.replan_period}s).")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def map_callback(self, msg: OccupancyGrid):
        """Just store the latest merged map; replanning happens on the timer."""
        self.latest_map = msg

    # ------------------------------------------------------------------
    # Robot pose via TF2
    # ------------------------------------------------------------------
    def get_robot_position(self, robot_name):
        """Look up robot's (x, y) in the merged 'map' frame."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', f'{robot_name}/base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            return np.array([t.transform.translation.x,
                             t.transform.translation.y])
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed for {robot_name}: {e}",
                throttle_duration_sec=10)
            return None

    # ------------------------------------------------------------------
    # Frontier detection (morphological dilation, not Canny)
    # ------------------------------------------------------------------
    def detect_frontiers(self, msg):
        """Return a list of (x, y) world-frame frontier centroids."""
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        grid = np.array(msg.data, dtype=np.int8).reshape((h, w))

        unknown_mask = np.where(grid == -1, 255, 0).astype(np.uint8)
        free_mask = np.where(grid == 0, 255, 0).astype(np.uint8)

        # Frontier = free cell adjacent to at least one unknown cell
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask, kernel, iterations=1)
        frontier_mask = cv2.bitwise_and(free_mask, dilated_unknown)

        num_labels, _labels, stats, centroids = \
            cv2.connectedComponentsWithStats(frontier_mask)

        points = []
        for i in range(1, num_labels):                      # skip background
            if stats[i, cv2.CC_STAT_AREA] > self.min_frontier_size:
                cx, cy = centroids[i]
                points.append(np.array([ox + cx * res,
                                        oy + cy * res]))
        return points

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(frontiers, radius):
        """Keep only one centroid per cluster within *radius* metres."""
        if not frontiers:
            return []
        unique = [frontiers[0]]
        for f in frontiers[1:]:
            if all(np.linalg.norm(f - u) > radius for u in unique):
                unique.append(f)
        return unique

    # ------------------------------------------------------------------
    # Timer-based replanning
    # ------------------------------------------------------------------
    def replan(self):
        if self.latest_map is None:
            return

        # 1. Detect & deduplicate frontiers on the merged map
        frontiers = self.detect_frontiers(self.latest_map)
        frontiers = self._deduplicate(frontiers, self.dedup_radius)
        self._publish_markers(frontiers)

        if not frontiers:
            self.get_logger().info(
                "Exploration complete – no frontiers remain.",
                throttle_duration_sec=30)
            return

        # 2. Get current robot positions
        positions = {}
        for name in self.robots:
            pos = self.get_robot_position(name)
            if pos is not None:
                positions[name] = pos

        if not positions:
            self.get_logger().warn(
                "No robot positions available, skipping replan.",
                throttle_duration_sec=10)
            return

        # 3. Decide which robots need a NEW goal (including stuck detection)
        robots_needing_goal = []
        for name in self.robots:
            if name not in positions:
                continue

            cur_pos = positions[name]
            last_pos = getattr(self, 'last_positions', {}).get(name)
            cur_goal = self.assigned_goals.get(name)

            if not hasattr(self, 'stuck_counters'):
                self.stuck_counters = {n: 0 for n in self.robots}
            if not hasattr(self, 'last_positions'):
                self.last_positions = {}

            # Check for stuck condition (has a goal, but moved < 0.15m in the last replan period)
            if cur_goal is not None and last_pos is not None:
                moved = np.linalg.norm(cur_pos - last_pos)
                if moved < 0.15:
                    self.stuck_counters[name] += 1
                else:
                    self.stuck_counters[name] = 0

                if self.stuck_counters[name] >= 1:
                    self.get_logger().info(
                        f"Robot {name} moved only {moved:.2f}m in {self.replan_period}s. Goal failed. Re-evaluating.")
                    self.assigned_goals[name] = None
                    self.stuck_counters[name] = 0
                    cur_goal = None

            self.last_positions[name] = cur_pos

            if cur_goal is not None:
                dist = np.linalg.norm(cur_pos - cur_goal)
                still_frontier = any(
                    np.linalg.norm(cur_goal - f) < self.dedup_radius
                    for f in frontiers)
                if dist > self.goal_reached_threshold and still_frontier:
                    continue                    # keep navigating to current goal
            robots_needing_goal.append(name)

        if not robots_needing_goal:
            return

        # 4. Remove frontiers already assigned to robots that are keeping
        #    their current goal (avoid sending two robots to the same spot)
        available = list(frontiers)
        for name in self.robots:
            if name not in robots_needing_goal:
                cur = self.assigned_goals.get(name)
                if cur is not None:
                    available = [f for f in available
                                 if np.linalg.norm(f - cur) > self.dedup_radius]

        # 5. Greedy nearest-frontier assignment (skip frontiers too close)
        for name in robots_needing_goal:
            if not available:
                break
            pos = positions[name]
            dists = [np.linalg.norm(pos - f) for f in available]

            # Filter: skip frontiers closer than min_goal_distance
            # (the robot is already there and has observed that area)
            candidates = [(d, i) for i, d in enumerate(dists)
                          if d > self.min_goal_distance]

            if not candidates:
                # All frontiers are nearby; fall back to the farthest one
                # to push the robot into truly new territory
                idx = int(np.argmax(dists))
            else:
                # Pick the nearest candidate that is beyond min_goal_distance
                candidates.sort()
                idx = candidates[0][1]

            target = available.pop(idx)
            self.assigned_goals[name] = target

            goal = PoseStamped()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = 'map'          # merged map frame
            goal.pose.position.x = float(target[0])
            goal.pose.position.y = float(target[1])
            goal.pose.orientation.w = 1.0

            self.pubs[name].publish(goal)
            self.get_logger().info(
                f"Assigned {name} -> ({target[0]:.2f}, {target[1]:.2f})")



    def _publish_markers(self, frontiers):
        marker_array = MarkerArray()
        
        # Delete old markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        if not frontiers:
            self.marker_pub.publish(marker_array)
            return

        for i, f in enumerate(frontiers):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'frontiers'
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(f[0])
            marker.pose.position.y = float(f[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MultiRobotFrontierCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
