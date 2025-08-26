#!/bin/bash
set -euo pipefail

# Should be running as root

# read user and group id from environment variables, or use defaults
USER_ID=${HOST_UID:-${USER_UID}}
GROUP_ID=${HOST_GID:-${USER_GID}}

# update user and group id if not matching
if [ "$(id -u $USERNAME)" != "$USER_ID" ]; then
    echo "Updating user UID to $USER_ID"
    usermod -u "$USER_ID" "$USERNAME"
fi
if [ "$(id -g $USERNAME)" != "$GROUP_ID" ]; then
    echo "Updating group GID to $GROUP_ID"
    groupmod -g "$GROUP_ID" "$USERNAME"
fi

# chown all files using find xargs chown
find /home/${USERNAME} \( -not -user $USER_ID -o -not -group $GROUP_ID \) -print0 | xargs -0 -P$(nproc) -n128 -r chown ${USERNAME}:${USERNAME}
find /app \( -not -user $USER_ID -o -not -group $GROUP_ID \) -print0 | xargs -0 -P$(nproc) -n128 -r chown ${USERNAME}:${USERNAME}

# switch to user
echo "Switching to user $USERNAME with UID $USER_ID and GID $GROUP_ID"
echo "Running command: $@"
exec gosu ${USERNAME} "$@"