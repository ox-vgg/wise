#!/bin/bash
## Copyright 2026 University of Oxford
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

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
find /home/${USERNAME} \( -not -user $USER_ID -o -not -group $GROUP_ID \) -print0 | xargs -0 -P$(nproc) -n128 -r chown ${USER_ID}:${GROUP_ID}
find /app \( -not -user $USER_ID -o -not -group $GROUP_ID \) -print0 | xargs -0 -P$(nproc) -n128 -r chown ${USER_ID}:${GROUP_ID}

# switch to user
echo "Switching to user $USERNAME with UID $USER_ID and GID $GROUP_ID"
echo "Running command: $@"
exec gosu ${USERNAME} "$@"
