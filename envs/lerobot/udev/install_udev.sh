#!/usr/bin/env bash
# Install the SO-101 udev rules (stable /dev/so101_* names) and apply them to devices already plugged in.
# Usage: envs/lerobot/udev/install_udev.sh    (asks for sudo)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES="99-so101.rules"

sudo install -m 0644 "$HERE/$RULES" "/etc/udev/rules.d/$RULES"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --subsystem-match=video4linux --action=change
sudo udevadm settle

echo "[install_udev] installed /etc/udev/rules.d/$RULES. Current links:"
ls -l /dev/so101_* 2>/dev/null || echo "  (none: is anything plugged in?)"
