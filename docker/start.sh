#!/bin/bash
# Pod entrypoint (CMD of docker/Dockerfile), standing in for runpod/base's
# /start.sh, which this Arch image no longer inherits: only the parts the pods
# use — sshd from the PUBLIC_KEY env var Runpod injects, and the pod env saved
# for SSH sessions. Keeps runpod's final log line, which `stream-pod-logs` is
# watched for.
set -e

if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
    # host keys per pod, not baked into the image
    ssh-keygen -A
    /usr/bin/sshd
fi

# Container ENV (Runpod's RUNPOD_* and NVIDIA_* included) reaches only PID 1:
# save it for SSH sessions, which source it from ~/.bashrc.
echo "Exporting environment variables..."
printenv | grep -E '^[A-Z_][A-Z0-9_]*=' | grep -v '^PUBLIC_KEY' \
    | awk -F = '{ val = $0; sub(/^[^=]*=/, "", val); print "export " $1 "=\"" val "\"" }' > /etc/rp_environment
grep -q 'source /etc/rp_environment' ~/.bashrc || echo 'source /etc/rp_environment' >> ~/.bashrc

echo "Start script(s) finished, Pod is ready to use."

sleep infinity
