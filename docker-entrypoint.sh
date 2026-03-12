#!/bin/sh
# When PULSE_SERVER is set (macOS Docker voice mode), configure ALSA
# to route audio through PulseAudio over TCP instead of direct hardware.
# On Linux with --device /dev/snd, PULSE_SERVER is unset and ALSA talks
# directly to hardware — this script is a no-op.
if [ -n "$PULSE_SERVER" ]; then
    cat > /etc/asound.conf <<EOF
pcm.!default {
    type pulse
}
ctl.!default {
    type pulse
}
EOF
fi

exec "$@"
