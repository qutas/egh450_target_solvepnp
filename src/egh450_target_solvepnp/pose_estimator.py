#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np
import tf2_ros
from geometry_msgs.msg import TransformStamped

from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class PoseEstimator():
	def __init__(self):
		# Set up the CV Bridge
		self.bridge = CvBridge()

		# Load in parameters from ROS
		self.param_use_compressed = rospy.get_param("~use_compressed", False)
		self.param_circle_radius = rospy.get_param("~circle_radius", 1.0)
		self.param_hue_center = rospy.get_param("~hue_center", 170)
		self.param_hue_range = rospy.get_param("~hue_range", 20) / 2
		self.param_sat_min = rospy.get_param("~sat_min", 50)
		self.param_sat_max = rospy.get_param("~sat_max", 255)
		self.param_val_min = rospy.get_param("~val_min", 50)
		self.param_val_max = rospy.get_param("~val_max", 255)

		# Set additional camera parameters
		self.got_camera_info = False
		self.camera_matrix = None
		self.dist_coeffs = None

		# Set up the publishers, subscribers, and tf2
		self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)

		if self.param_use_compressed:
			self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image_raw/compressed", CompressedImage, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
		else:
			self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image_raw", Image, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

		self.tfbr = tf2_ros.TransformBroadcaster()

		# Generate the model for the pose solver
		# For this example, draw a square around where the circle should be
		# There are 5 points, one in the center, and one in each corner
		r = self.param_circle_radius
		self.model_object = np.array([(0.0, 0.0, 0.0),
										(r, r, 0.0),
										(r, -r, 0.0),
										(-r, r, 0.0),
										(-r, -r, 0.0)])

	def shutdown(self):
		# Unregister anything that needs it here
		self.sub_info.unregister()
		self.sub_img.unregister()

	# Collect in the camera characteristics
	def callback_info(self, msg_in):
		self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")

		self.camera_matrix = np.array([
                 (msg_in.P[0], msg_in.P[1], msg_in.P[2]),
                 (msg_in.P[4], msg_in.P[5], msg_in.P[6]),
                 (msg_in.P[8], msg_in.P[9], msg_in.P[10])],
				 dtype="double")

		if not self.got_camera_info:
			rospy.loginfo("Got camera info")
			self.got_camera_info = True

	def callback_img(self, msg_in):
		# Don't bother to process image if we don't have the camera calibration
		if self.got_camera_info:
			#Convert ROS image to CV image
			cv_image = None

			try:
				if self.param_use_compressed:
					cv_image = self.bridge.compressed_imgmsg_to_cv2( msg_in, "bgr8" )
				else:
					cv_image = self.bridge.imgmsg_to_cv2( msg_in, "bgr8" )
			except CvBridgeError as e:
				rospy.loginfo(e)
				return

			# Perform a colour mask for detection
			mask_image = self.process_image(cv_image)

			# Find circles in the masked image
			min_dist = mask_image.shape[0]/8
			circles = cv2.HoughCircles(mask_image, cv2.HOUGH_GRADIENT, 1, min_dist, param1=50, param2=20, minRadius=0, maxRadius=0)

			# If circles were detected
			if circles is not None:
				# Just take the first detected circle
				px = circles[0,0,0]
				py = circles[0,0,1]
				pr = circles[0,0,2]

				# Calculate the pictured the model for the pose solver
				# For this example, draw a square around where the circle should be
				# There are 5 points, one in the center, and one in each corner
				self.model_image = np.array([
											(px, py),
											(px+pr, py+pr),
											(px+pr, py-pr),
											(px-pr, py+pr),
											(px-pr, py-pr)])

				# Do the SolvePnP method
				(success, rvec, tvec) = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)

				# If a result was found, send to TF2
				if success:
					msg_out = TransformStamped()
					msg_out.header = msg_in.header
					msg_out.child_frame_id = "circle"
					msg_out.transform.translation.x = tvec[0]
					msg_out.transform.translation.y = tvec[1]
					msg_out.transform.translation.z = tvec[2]
					msg_out.transform.rotation.w = 1.0	# Could use rvec, but need to convert from DCM to quaternion first
					msg_out.transform.rotation.x = 0.0
					msg_out.transform.rotation.y = 0.0
					msg_out.transform.rotation.z = 0.0

					self.tfbr.sendTransform(msg_out)

				# Draw the circle for the overlay
				cv2.circle(cv_image, (px,py), 2, (255, 0, 0), 2)	# Center
				cv2.circle(cv_image, (px,py), pr, (0, 0, 255), 2)	# Outline
				cv2.rectangle(cv_image, (px-pr,py-pr), (px+pr,py+pr), (0, 255, 0), 2)	# Model

			#Convert CV image to ROS image and publish the mask / overlay
			try:
				if self.param_use_compressed:
					self.pub_mask.publish( self.bridge.cv2_to_compressed_imgmsg( mask_image, "png" ) )
					self.pub_overlay.publish( self.bridge.cv2_to_compressed_imgmsg( cv_image, "png" ) )
				else:
					self.pub_mask.publish( self.bridge.cv2_to_imgmsg( mask_image, "mono8" ) )
					self.pub_overlay.publish( self.bridge.cv2_to_imgmsg( cv_image, "bgr8" ) )
			except (CvBridgeError,TypeError) as e:
				rospy.loginfo(e)

	def process_image(self, cv_image):
		#Convert the image to HSV and prepare the mask
		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		mask_image = None

		hue_lower = (self.param_hue_center - self.param_hue_range) % 180
		hue_upper = (self.param_hue_center + self.param_hue_range) % 180

		thresh_lower = np.array([hue_lower, self.param_val_min, self.param_val_min])
		thresh_upper = np.array([hue_upper, self.param_val_max, self.param_val_max])


		if hue_lower > hue_upper:
			# We need to do a wrap around HSV 180 to 0 if the user wants to mask this color
			thresh_lower_wrap = np.array([180, self.param_sat_max, self.param_val_max])
			thresh_upper_wrap = np.array([0, self.param_sat_min, self.param_val_min])

			mask_lower = cv2.inRange(hsv_image, thresh_lower, thresh_lower_wrap)
			mask_upper = cv2.inRange(hsv_image, thresh_upper_wrap, thresh_upper)

			mask_image = cv2.bitwise_or(mask_lower, mask_upper)
		else:
			# Otherwise do a simple mask
			mask_image = cv2.inRange(hsv_image, thresh_lower, thresh_upper)

		# Refine image to get better results
		kernel = np.ones((5,5),np.uint8)
		mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

		return mask_image









