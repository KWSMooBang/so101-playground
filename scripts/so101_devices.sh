# shellcheck shell=bash
# Device paths and camera spec for the home-desktop SO-101, sourced by teleop.sh and record.sh.
# /dev/so101_* once envs/lerobot/udev/install_udev.sh has run, else the by-id/by-path links.

pick() { [ -e "$1" ] && echo "$1" || echo "$2"; }
LEADER=$(pick /dev/so101_leader /dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC452337-if00)
FOLLOWER=$(pick /dev/so101_follower /dev/serial/by-id/usb-1a86_USB_Single_Serial_5CBC453103-if00)
CAM_WRIST=$(pick /dev/so101_cam_wrist /dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.4.2:1.0-video-index0)
CAM_OVERHEAD=$(pick /dev/so101_cam_overhead /dev/v4l/by-path/pci-0000:00:14.0-usb-0:6.4.3:1.0-video-index0)

cam() { echo "{type: opencv, index_or_path: $1, width: 640, height: 480, fps: 30, fourcc: MJPG}"; }

# Arms in the SO-101 Bench datasets' units: use_degrees=false (RANGE_M100_100, gripper 0..100).
ARM_ARGS=(
    --robot.type=so101_follower
    --robot.port="$FOLLOWER"
    --robot.id=so101_follower_arm
    --robot.use_degrees=false
    --robot.cameras="{wrist: $(cam "$CAM_WRIST"), overhead: $(cam "$CAM_OVERHEAD")}"
    --teleop.type=so101_leader
    --teleop.port="$LEADER"
    --teleop.id=so101_leader_arm
    --teleop.use_degrees=false
)
