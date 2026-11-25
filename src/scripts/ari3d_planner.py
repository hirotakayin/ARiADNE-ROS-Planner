#!/usr/bin/env python3
"""The learned 3D exploration planner (ARiADNE 3D, docs/PLAN_3D.md in hirotakayin/large-scale-DRL-exploration) on ROS.

It runs the training code itself (env3d.Map3D, node_manager3d, agent3d, model; ~training_dir, a checkout of branch
3d-learning) on a voxel map it builds from the lidar, so the policy sees the same inputs as in training:

  map        Map3D at the training resolution (CELL_SIZE, 0.4 m) over a fixed box: ~course (an evaluation course
             folder from voxel_world's make_courses.py: course.npz gives the box and the start) or ~map_origin,
             ~map_size and ~start (world metres). Every cloud (/cloud_registered_body, lidar frame, through TF into
             ~frame) is cast in: voxels along each ray up to SENSOR_RANGE become free, the voxel of a return within
             range occupied (occupied stays occupied). Rays the simulated Mid-360 reports as points at the sensor
             (no return) carry no direction and are dropped.
  decision   only when asked (service /ari3d/next, called by ari3d_mission.py when the drone reached the last target),
             at the lattice node it was sent to, like a stop in training: the drone's clearance box counts as free,
             cfree and frontiers are recomputed, the dead-frontier count advances (looked_from), the graph is updated,
             and the policy (argmax) picks a neighbouring column; ~fallback takes the greedy explorer's first hop when
             no neighbour has utility or the policy turns back (check_env3d.py --fallback). LEFTOVER_M is ~leftover_m.
             Unlike training the map also grows between stops, since the lidar runs all the time.
  answer     Trigger response, message = JSON: {"done": bool, "route": [[x, y, z], ...] (world metres, lattice
             nodes after the current one), "target": [x, y, z], "fallback": bool, "plan_time": s, "reason": str}.

Publishes /ari3d/start (latched PointStamped: the lattice node to take off to), /ari3d/route (Path), /ari3d/status
(JSON at 1 Hz: decisions, explored volume % when the course's ground truth is known, utility, plan time, distance
flown) and /ari3d/columns (Marker: columns with utility, green if they count). Writes a run record to
~run_dir/<time>_<name>.json when exploration ends and on shutdown.
"""
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger, TriggerResponse
from visualization_msgs.msg import Marker


def expand(p):
    return os.path.abspath(os.path.expanduser(p))


def setup_training_code():
    """Put the training code and voxel_world on the path and set the environment parameter.py reads."""
    training = expand(rospy.get_param('~training_dir', '~/rl/ari3d_deploy'))
    voxel = expand(rospy.get_param('~voxel_world_src', '~/ws_3d/src/CERLAB-UAV-Autonomy/voxel_world/src'))
    for d, what in ((training, 'parameter.py'), (os.path.join(voxel, 'voxel_world'), 'fast.py')):
        if not os.path.exists(os.path.join(d, what)):
            rospy.logfatal('[ari3d] missing %s in %s' % (what, d))
            sys.exit(1)
    os.environ['ENV_DIM'] = '3'
    os.environ['LEFTOVER_M'] = str(float(rospy.get_param('~leftover_m', 0.0)))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    sys.path[:0] = [training, voxel]
    return training


def cloud_xyz(msg):
    """(N, 3) float64 x y z of a PointCloud2 with float32 x, y, z fields."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4'] * 3,
                   'offsets': [off['x'], off['y'], off['z']], 'itemsize': msg.point_step})
    a = np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)
    xyz = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Planner:
    def __init__(self):
        self.training_dir = setup_training_code()
        global torch, P, Map3D, FREE3, OCC3, UNKNOWN3, Agent3D, PolicyNet, load_widened, traverse
        import torch
        import parameter as P
        from env3d import Map3D, FREE3, OCC3, UNKNOWN3
        from agent3d import Agent3D
        from model import PolicyNet, load_widened
        from voxel_world.fast import traverse
        torch.set_num_threads(int(rospy.get_param('~torch_threads', 2)))

        self.frame = rospy.get_param('~frame', 'map')
        self.fallback = bool(rospy.get_param('~fallback', False))
        self.max_decisions = int(rospy.get_param('~max_decisions', 300))
        self.min_range = float(rospy.get_param('~min_range', 0.3))
        self.res = float(P.CELL_SIZE)
        self.range = float(P.SENSOR_RANGE)
        self.lock = threading.Lock()

        # the box, the start and (for a course) the ground truth used for the explored volume
        self.name = rospy.get_param('~name', '')
        self.truth = None
        course = rospy.get_param('~course', '')
        if course:
            z = np.load(os.path.join(expand(course), 'course.npz'))
            origin, size, start = z['origin'].astype(float), z['extent'].astype(float), z['start'].astype(float)
            self.name = self.name or os.path.basename(expand(course).rstrip('/'))
            r = float(z['resolution'])
            kz, ky, kx = np.nonzero(z['explorable'])
            self.truth = np.stack([kx, ky, kz], 1) * r + r / 2 + origin  # world metres
            # solid grown by one voxel: a surface can sit inside a voxel whose centre is free (the course
            # rasterizes boxes by voxel centre), and the lidar then marks the map cell holding it occupied
            from scipy import ndimage
            oz, oy, ox = np.nonzero(ndimage.binary_dilation(z['occupied'], np.ones((3, 3, 3), bool)))
            self.truth_solid = np.stack([ox, oy, oz], 1) * r + r / 2 + origin
        else:
            origin = np.array(rospy.get_param('~map_origin'), float)
            size = np.array(rospy.get_param('~map_size'), float)
            start = np.array(rospy.get_param('~start'), float)
        self.origin = origin
        shape_xyz = np.ceil(size / self.res).astype(int)
        known = np.zeros(shape_xyz[::-1], np.int8)
        start_node = (np.floor((start - origin) / self.res) + 0.5) * self.res  # grid metres, a voxel centre
        self.map = Map3D(known, self.res, start_node, name=self.name)
        self.node = start_node.copy()  # the lattice node the drone is at (or was last sent to), grid metres
        if self.truth is not None:
            # the map's cells (CELL_SIZE) that hold explorable space and no solid: the ones it can know as free
            cells = np.unique(np.floor((self.truth - origin) / self.res).astype(int), axis=0)
            solid = np.unique(np.floor((self.truth_solid - origin) / self.res).astype(int), axis=0)
            ok = np.all((cells >= 0) & (cells < shape_xyz), axis=1)
            cells = cells[ok]
            mixed = (cells[:, None, :] == solid[None]).all(-1).any(1) if len(cells) * len(solid) < 3e7 else \
                np.isin(cells @ [1, 10 ** 4, 10 ** 8], solid @ [1, 10 ** 4, 10 ** 8])
            self.truth_idx = cells[~mixed]
            self.truth_total = len(self.truth_idx)

        ckpt = expand(rospy.get_param('~checkpoint'))
        self.net = PolicyNet(P.NODE_INPUT_DIM, P.EMBEDDING_DIM)
        load_widened(self.net, torch.load(ckpt, map_location='cpu', weights_only=False)['policy_model'])
        self.net.eval()
        self.agent = Agent3D(self.net, 'cpu')
        self.agent.fallback_hist = []
        self.checkpoint = ckpt

        self.decisions = []
        self.done = False
        self.done_reason = ''
        self.started = None
        self.flown = 0.0
        self.last_odom = None
        self.clouds = 0
        self.cloud_time = 0.0
        self.saved = False

        self.tf = tf2_ros.Buffer(rospy.Duration(10.0))
        tf2_ros.TransformListener(self.tf)
        self.start_pub = rospy.Publisher('/ari3d/start', PointStamped, queue_size=1, latch=True)
        self.route_pub = rospy.Publisher('/ari3d/route', Path, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/ari3d/status', String, queue_size=1, latch=True)
        self.cols_pub = rospy.Publisher('/ari3d/columns', Marker, queue_size=1, latch=True)
        rospy.Subscriber(rospy.get_param('~cloud_topic', '/cloud_registered_body'), PointCloud2, self.cloud_cb,
                         queue_size=1, buff_size=2 ** 24)
        rospy.Subscriber(rospy.get_param('~odom_topic', '/CERLAB/quadcopter/odom'), Odometry, self.odom_cb,
                         queue_size=10)
        rospy.Service('/ari3d/next', Trigger, self.next_srv)

        s = PointStamped()
        s.header.frame_id = self.frame
        s.header.stamp = rospy.Time.now()
        s.point.x, s.point.y, s.point.z = (start_node + origin).tolist()
        self.start_pub.publish(s)
        rospy.Timer(rospy.Duration(1.0), self.status_tick)
        rospy.on_shutdown(lambda: self.save('shutdown'))
        rospy.loginfo('[ari3d] %s: grid %s at %.1f m, start node %s, checkpoint %s, LEFTOVER_M %s, fallback %s' % (
            self.name, shape_xyz.tolist(), self.res, np.round(start_node + origin, 2).tolist(), ckpt, P.LEFTOVER_M,
            self.fallback))

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        p = np.array([p.x, p.y, p.z])
        if self.last_odom is not None and self.started is not None and not self.done:
            self.flown += float(np.linalg.norm(p - self.last_odom))
        self.last_odom = p

    def cloud_cb(self, msg):
        try:
            t = self.tf.lookup_transform(self.frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            return
        tic = time.time()
        pts = cloud_xyz(msg)
        pts = pts[np.linalg.norm(pts, axis=1) > self.min_range]
        if len(pts) == 0:
            return
        R = quat_matrix(t.transform.rotation)
        tr = t.transform.translation
        sensor = np.array([tr.x, tr.y, tr.z])
        world = pts @ R.T + sensor
        with self.lock:
            self.integrate(sensor - self.origin, world - self.origin)
        self.clouds += 1
        self.cloud_time += time.time() - tic

    def integrate(self, s, ends):
        """Cast one cloud into the map (grid metres): free along each ray, occupied at a return within range."""
        known, res = self.map.map, self.res
        shape = self.map.shape_xyz
        if not np.all((s >= 0) & (s < shape * res)):
            return
        d = ends - s
        r = np.linalg.norm(d, axis=1)
        hit = r <= self.range
        ends = np.where(hit[:, None], ends, s + d / r[:, None] * self.range)
        ev = np.floor(ends / res).astype(np.int64)
        _, keep = np.unique(ev, axis=0, return_index=True)  # one ray per end voxel
        ends, hit, ev = ends[keep], hit[keep], ev[keep]
        ray, x, y, z = traverse(s, ends, np.zeros(3), res, shape)
        if len(ray) == 0:
            return
        is_end = (x == ev[ray, 0]) & (y == ev[ray, 1]) & (z == ev[ray, 2])
        occ = is_end & hit[ray]
        f = ~occ
        cur = known[z[f], y[f], x[f]]
        known[z[f], y[f], x[f]] = np.where(cur == OCC3, OCC3, FREE3)
        known[z[occ], y[occ], x[occ]] = OCC3

    # ------------------------------------------------------------------ decisions
    def next_srv(self, _):
        if self.done:
            return TriggerResponse(True, json.dumps({'done': True, 'reason': self.done_reason}))
        try:
            with self.lock:
                out = self.decide()
        except Exception as e:  # report it to the mission rather than dying silently
            rospy.logerr('[ari3d] decision failed: %r' % (e,))
            return TriggerResponse(False, json.dumps({'done': False, 'reason': 'decision failed: %r' % (e,)}))
        if out.get('done'):
            self.save('done')
        return TriggerResponse(True, json.dumps(out))

    def decide(self):
        tic = time.time()
        if self.started is None:
            self.started = time.time()
        p = self.node.copy()
        known, res = self.map.map, self.res
        i = np.floor(p / res).astype(int)
        c = max(1, P.CLEARANCE_3D)
        box = tuple(slice(max(0, i[k] - c), i[k] + c + 1) for k in (2, 1, 0))
        sub = known[box]
        sub[sub == UNKNOWN3] = FREE3  # the drone's own clearance box (the band never sees straight down)
        self.map.robot_position = p.copy()
        self.map.refresh()
        self.map.looked_from(p)
        ag = self.agent
        ag.update_planning_state(self.map, self.map.local_xy(p))
        nm = ag.node_manager
        explored = self.explored()
        if ag.utility.sum() == 0 or len(self.decisions) >= self.max_decisions:
            self.done = True
            self.done_reason = 'no column has utility' if ag.utility.sum() == 0 else \
                'decision limit %d' % self.max_decisions
            self.publish_columns()
            return {'done': True, 'reason': self.done_reason, 'explored': explored}
        obs = ag.get_observation()
        _, _, _, _, current_edge, _ = obs
        with torch.no_grad():
            logp = self.net(*obs)
        k = int(torch.argmax(logp, dim=1).item())
        used_fallback = False
        if self.fallback:
            k2 = fallback_action(ag, current_edge, k)
            if k2 is not None:
                k, used_fallback = k2, True
        ag.fallback_hist.append(tuple(ag.node_coords[ag.current_index]))
        nxt = ag.node_coords[int(current_edge[0, k, 0].item())]
        target, length, path = nm.route_to(nxt)
        route = [(nm.lat.pos[j] + self.origin).tolist() for j in path[1:]]
        self.node = np.asarray(target, float).copy()
        plan_time = time.time() - tic
        rec = dict(i=len(self.decisions), t=round(time.time() - self.started, 2),
                   at=np.round(p + self.origin, 2).tolist(), target=np.round(target + self.origin, 2).tolist(),
                   route_m=round(float(length), 2), fallback=used_fallback, p=round(float(logp[0, k].exp()), 3),
                   utility=round(float(ag.utility.sum()), 2), columns_with_utility=int((ag.utility > 0).sum()),
                   plan_time=round(plan_time, 3), explored=explored, flown=round(self.flown, 1))
        self.decisions.append(rec)
        self.publish_route(route)
        self.publish_columns()
        rospy.loginfo('[ari3d] decision %d: %s -> %s, route %.1f m%s, plan %.2f s, explored %s' % (
            rec['i'], rec['at'], rec['target'], length, ' (fallback)' if used_fallback else '', plan_time,
            '-' if explored is None else '%.1f %%' % explored))
        return {'done': False, 'route': route, 'target': rec['target'], 'fallback': used_fallback,
                'plan_time': plan_time, 'route_m': length}

    def explored(self):
        """Explored volume: % of the map cells holding explorable space and no solid (the course's ground truth) that
        the map knows as free (training's metric, on the cells a CELL_SIZE map can know as free)."""
        if self.truth is None:
            return None
        ti = self.truth_idx
        free = self.map.map[ti[:, 2], ti[:, 1], ti[:, 0]] == FREE3
        return round(100.0 * free.sum() / self.truth_total, 2)

    # ------------------------------------------------------------------ outputs
    def publish_route(self, route):
        msg = Path()
        msg.header.frame_id = self.frame
        msg.header.stamp = rospy.Time.now()
        for q in [(self.decisions[-1]['at'] if self.decisions else None)] + route:
            if q is None:
                continue
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = q
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.route_pub.publish(msg)

    def publish_columns(self):
        nm = self.agent.node_manager
        if nm.lat is None:
            return
        m = Marker()
        m.header.frame_id = self.frame
        m.header.stamp = rospy.Time.now()
        m.ns, m.id, m.type, m.action = 'ari3d_columns', 0, Marker.SPHERE_LIST, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.pose.orientation.w = 1.0
        for node in nm.nodes_dict:
            d = node.data
            if d.kept < 0 or nm.util[d.kept] == 0:
                continue
            q = nm.lat.pos[d.kept] + self.origin
            m.points.append(Point(*q.tolist()))
            m.colors.append(ColorRGBA(0.1, 0.7, 0.3, 1.0) if d.utility > 0 else ColorRGBA(0.6, 0.6, 0.6, 0.6))
        self.cols_pub.publish(m)

    def status_tick(self, _):
        last = self.decisions[-1] if self.decisions else {}
        self.status_pub.publish(String(json.dumps(dict(
            name=self.name, decisions=len(self.decisions), done=self.done, reason=self.done_reason,
            explored=self.explored(), flown=round(self.flown, 1),
            seconds=round(time.time() - self.started, 1) if self.started else 0.0,
            fallbacks=sum(d['fallback'] for d in self.decisions), last_plan_time=last.get('plan_time'),
            utility=last.get('utility'), clouds=self.clouds,
            cloud_ms=round(1000 * self.cloud_time / max(1, self.clouds), 1)))))

    def save(self, why):
        if self.saved or not self.decisions:
            return
        self.saved = True
        d = expand(rospy.get_param('~run_dir', '~/mission_runs/ari3d_runs'))
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, time.strftime('%Y%m%d_%H%M%S') + '_%s.json' % (self.name or 'run'))
        np.savez_compressed(f[:-5] + '_map.npz', map=self.map.map, frontier=self.map.frontier, dead=self.map.dead,
                            origin=self.origin, res=self.res, start=self.map.start,
                            truth_cells=self.truth_idx if self.truth is not None else np.zeros((0, 3), int))
        json.dump(dict(name=self.name, why=why, done=self.done, reason=self.done_reason, checkpoint=self.checkpoint,
                       training_dir=self.training_dir, leftover_m=P.LEFTOVER_M, fallback=self.fallback,
                       explored=self.explored(), flown=round(self.flown, 1),
                       seconds=round(time.time() - self.started, 1) if self.started else None,
                       decisions=self.decisions), open(f, 'w'), indent=1)
        rospy.loginfo('[ari3d] run record: %s' % f)


def fallback_action(agent, current_edge, k):
    """check_env3d.py's --fallback: when no neighbouring column has utility, or the policy's choice is the stop it
    came from, the first hop of the route to the greedy explorer's target (most utility per metre of route).
    Returns the action index to use instead of k, or None to keep k."""
    u = np.asarray(agent.utility, float)
    if u.sum() <= 0:
        return None
    edges = [int(current_edge[0, i, 0].item()) for i in range(current_edge.shape[1])]
    nbrs = [j for j in agent.neighbor_indices if j != agent.current_index]
    hist = agent.fallback_hist
    back = len(hist) >= 1 and tuple(agent.node_coords[edges[k]]) == hist[-1]
    if any(u[j] > 0 for j in nbrs) and not back:
        return None
    nm = agent.node_manager
    kept = [nm.nodes_dict.find(c.tolist()).data.kept for c in agent.node_coords]
    d = nm.dist[kept]
    score = np.where(u > 0, u / (d + P.NODE_RESOLUTION), -1)
    score[agent.current_index] = -1
    v = kept[int(np.argmax(score))]
    hops = []
    while v != nm.here and v >= 0:
        hops.append(v)
        v = nm.pred[v]
    here_col = nm.lat.col[nm.here]
    first = next((h for h in reversed(hops) if nm.lat.col[h] != here_col), None)
    if first is None:
        return None
    col = nm.lat.col[first]
    for i, j in enumerate(edges):
        if j in nbrs:
            kk = nm.nodes_dict.find(agent.node_coords[j].tolist()).data.kept
            if kk >= 0 and nm.lat.col[kk] == col:
                return i
    return None


if __name__ == '__main__':
    rospy.init_node('ari3d_planner')
    Planner()
    rospy.spin()
