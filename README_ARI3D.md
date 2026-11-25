# ARiADNE 3D on ROS (branch 3d-learning)

The learned 3D exploration planner from hirotakayin/large-scale-DRL-exploration (branch `3d-learning`,
`docs/PLAN_3D.md`) flying in Gazebo. The 2D planner (`rl_planner.py`, the authors' code) is untouched.

| File | What it is |
|---|---|
| `src/scripts/ari3d_planner.py` | Builds a voxel map from the lidar and runs the training code's `Map3D`, `NodeManager3D`, `Agent3D` and policy on it. Decides on request (`/ari3d/next`). |
| `src/scripts/ari3d_mission.py` | Takes off to the planner's start node, flies each route, asks for the next one on arrival, flies home along the flown path and lands. |
| `src/scripts/ari3d_course_world.py` | A course's Gazebo world with its room inlined (`<course>/world_inline.sdf`), and its spawn pose. |
| `src/scripts/ari3d_course.sh` | Starts the launch file on one course with the spawn pose read from it. |
| `src/launch/ari3d_course.launch` | Gazebo with the course, the drone (`quadcopter_lidar.urdf`), the tracking controller, planner and mission. |
| `src/rviz/ari3d.rviz` | Lidar, route, columns with utility (`rviz:=true`). |

## Run it

```
source ~/catkin_ws/devel/setup.bash        # or ~/ws_3d: rl_planner is the same package in both
rosrun rl_planner ari3d_course.sh ~/mission_runs/courses/test_v2/course_0_floor [sim_gui:=true] [rviz:=true]
rosservice call /exploration/start
```

Launch arguments: `checkpoint` (default `~/rl/ari3d_checkpoints/stage1_25d/checkpoint.pth`), `leftover_m` (20),
`fallback` (false), `max_decisions` (300), `speed` (1.0 m/s), `max_seconds` (1200), `lidar_pitch_deg` (23.5),
`run_dir` (`~/mission_runs/ari3d_runs`), `training_dir` (`~/rl/ari3d_deploy`), `voxel_world_src`, `python`.

Needs:
- `~/.venvs/ariadne`: torch, scipy, scikit-image, imageio. It uses the system site packages, so rospy imports.
- The training code at `training_dir`, a checkout of `3d-learning`. On the laptop that is a git worktree of
  `~/rl/large-scale-DRL-exploration`, detached at the commit the checkpoint was trained with; move it with
  `git -C ~/rl/ari3d_deploy checkout --detach <commit>`.
- voxel_world's `src` (for `voxel_world.fast.traverse`).
- Floor-start courses only: the drone spawns on the floor under the course's start.

## How it matches training

- **Map:** `Map3D` at `CELL_SIZE` (0.4 m) over the course's box. Each `/cloud_registered_body` cloud goes through TF into
  `map`. Voxels along a ray up to `SENSOR_RANGE` (16 m) become free; the voxel of a return within range becomes
  occupied, and occupied stays occupied. The simulated Mid-360 reports rays with no return as points at the sensor;
  they carry no direction and are dropped.
- **Decisions:** made only at a lattice node, like a stop in training. The drone's clearance box counts as free, then
  cfree and frontiers are recomputed and the dead-frontier count advances. The policy takes the argmax over the
  neighbouring columns. `LEFTOVER_M` and the fallback are the training repo's.
- **Difference from training:** the lidar runs all the time, so the map also grows between stops. In training it
  only grows at stops.

## Outputs

- `/ari3d/start`: latched PointStamped, the lattice node to take off to.
- `/ari3d/route`: Path.
- `/ari3d/columns`: Marker, the columns with utility; green ones pass `LEFTOVER_M`.
- `/ari3d/status`: JSON at 1 Hz. `explored` is the % of the map cells that hold explorable space and no solid (from
  `course.npz`) that the map knows as free. Also `decisions`, `fallbacks`, `flown`, `seconds`, `last_plan_time`,
  `utility`, `clouds`, `cloud_ms`, `done`, `reason`.
- `/ari3d_mission/state`: IDLE, TAKEOFF, EXPLORING, RETURNING, LANDING, LANDED.
- `/ari3d_mission/event`, `/ari3d_mission/plan`: as `ariadne_mission.py`'s.
- `/exploration/{start,return_home,land}`: Trigger services. Pause and resume are refused.
- Run record: `<run_dir>/<time>_<course>.json` (every decision: position, target, route length, fallback, policy
  probability, utility, plan time, explored, flown), and `<time>_<course>_map.npz` (final map, frontiers, dead
  frontiers, the cells counted for `explored`).
