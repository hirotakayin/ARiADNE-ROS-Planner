#!/bin/bash
# Start ari3d_course.launch on one evaluation course with the spawn pose read from the course.
# usage: rosrun rl_planner ari3d_course.sh <course dir> [arg:=value ...]
#   e.g. rosrun rl_planner ari3d_course.sh ~/mission_runs/courses/test_v2/course_0_floor leftover_m:=20 sim_gui:=true
set -euo pipefail
course=$(readlink -f "${1:?usage: ari3d_course.sh <course dir> [arg:=value ...]}"); shift
here=$(dirname "$(readlink -f "$0")")
py=${ARI3D_PYTHON:-$HOME/.venvs/ariadne/bin/python}
read -r x y z < <("$py" "$here/ari3d_course_world.py" "$course" --spawn)
exec roslaunch rl_planner ari3d_course.launch course:="$course" spawn_x:="$x" spawn_y:="$y" spawn_z:="$z" "$@"
