std_msgs/Header header
sensor_msgs/Image image  # Optional, if you want to include the original image
int32[] original_image_shape  # [height, width]
BoundingBox[] boxes