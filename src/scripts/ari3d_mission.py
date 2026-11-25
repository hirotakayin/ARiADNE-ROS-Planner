#!/usr/bin/env python3
"""Flies ari3d_planner.py's routes in the simulator: takes off to the planner's start node, then asks for the next route
(/ari3d/next) each time it reaches the end of the last one, flies it through the lattice nodes at the mission speed
(nose into the direction of travel, so the tilted Mid-360 looks ahead), and when the planner says exploration is done
(or on /exploration/return_home, or after ~max_seconds) retraces the flown path home and lands where it started.

Modelled on autonomous_flight's ariadne_mission.py (the 2D baseline): the same tracking-controller target, takeoff and
land topics, /exploration/* services and state / event / plan topics (here under /ari3d_mission/), so the dashboard
and the scoring monitor can read it the same way.
"""
import heapq
import json
import math
import threading

import rospy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger, TriggerResponse
from tracking_controller.msg import Target

RATE = 50.0
CRUMB_SPACING = 0.25  # m between remembered positions on the flown path
CRUMB_LINK = 0.8  # m, remembered positions this close count as connected on the way home
FACE_TRAVEL = 0.3  # m, turn the nose towards the next node only while it is at least this far away
LEASH = 1.0  # m, the setpoint waits while it is this far ahead of the drone
ARRIVED = 0.35  # m from the last route node that counts as there (the planner then decides from that node)
PLAN_RETRY = 1.0  # s between planner calls after a failed one
PLAN_FAILURES = 5  # failed planner calls in a row that end the exploration


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def dist(a, b):
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


class Mission:
    def __init__(self):
        self.home = rospy.get_param('~home')  # where the sim spawned the drone
        self.speed = rospy.get_param('~speed', 1.0)
        self.yaw_rate = rospy.get_param('~yaw_rate', 1.0)
        self.max_seconds = rospy.get_param('~max_seconds', 1200.0)

        self.state = 'IDLE'
        self.position = None
        self.odom_yaw = 0.0
        self.setpoint = list(self.home)
        self.yaw = self.yaw_target = 0.0
        self.route = []
        self.goal = None
        self.crumbs = []
        self.start_node = None
        self.fly_z = None
        self.explore_since = None
        self.pending = False
        self.answer = None
        self.failures = 0
        self.retry_at = rospy.Time(0)

        self.target_pub = rospy.Publisher('/autonomous_flight/target_state', Target, queue_size=10)
        self.takeoff_pub = rospy.Publisher(rospy.get_param('~takeoff_topic', '/CERLAB/quadcopter/takeoff'), Empty,
                                           queue_size=1, latch=True)
        self.land_pub = rospy.Publisher(rospy.get_param('~land_topic', '/CERLAB/quadcopter/land'), Empty,
                                        queue_size=1, latch=True)
        self.state_pub = rospy.Publisher('/ari3d_mission/state', String, queue_size=10, latch=True)
        self.event_pub = rospy.Publisher('/ari3d_mission/event', String, queue_size=50)
        self.plan_pub = rospy.Publisher('/ari3d_mission/plan', String, queue_size=50)
        self.path_pub = rospy.Publisher('/dynamicExploration/bspline_trajectory', Path, queue_size=1)

        rospy.Subscriber(rospy.get_param('~odom_topic', '/CERLAB/quadcopter/odom'), Odometry, self.odom_cb,
                         queue_size=10)
        rospy.Subscriber('/ari3d/start', PointStamped, self.start_cb, queue_size=1)
        rospy.Service('/exploration/start', Trigger, self.start_srv)
        rospy.Service('/exploration/return_home', Trigger, self.return_srv)
        rospy.Service('/exploration/land', Trigger, self.land_srv)
        rospy.Service('/exploration/pause', Trigger, self.no_pause_srv)
        rospy.Service('/exploration/resume', Trigger, self.no_pause_srv)
        self.next = rospy.ServiceProxy('/ari3d/next', Trigger)

        self.set_state('IDLE')
        rospy.Timer(rospy.Duration(1.0 / RATE), self.tick)

    def set_state(self, state):
        self.state = state
        self.state_pub.publish(String(state))

    def event(self, text):
        rospy.loginfo('[ari3d_mission] ' + text)
        self.event_pub.publish(String(text))

    # ------------------------------------------------------------------ inputs
    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = (p.x, p.y, p.z)
        self.odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.state not in ('IDLE', 'LANDED') and (not self.crumbs or dist(self.crumbs[-1], self.position) >= CRUMB_SPACING):
            self.crumbs.append(self.position)

    def start_cb(self, msg):
        self.start_node = (msg.point.x, msg.point.y, msg.point.z)
        self.fly_z = msg.point.z

    # ------------------------------------------------------------------ dashboard buttons
    def start_srv(self, _):
        if self.state != 'IDLE':
            return TriggerResponse(False, 'Mission already started (%s)' % self.state)
        if self.position is None:
            return TriggerResponse(False, 'No odometry yet')
        if self.start_node is None:
            return TriggerResponse(False, 'No start node from ari3d_planner yet')
        self.setpoint = list(self.position)
        self.yaw = self.yaw_target = self.odom_yaw
        self.crumbs = [self.position]
        self.route = [(self.position[0], self.position[1], self.fly_z), self.start_node]
        self.takeoff_pub.publish(Empty())
        self.set_state('TAKEOFF')
        self.event('Taking off to the start node %s' % (tuple(round(v, 2) for v in self.start_node),))
        return TriggerResponse(True, 'Mission started')

    def return_srv(self, _):
        if self.state != 'EXPLORING':
            return TriggerResponse(False, 'Cannot return home now (%s)' % self.state)
        self.start_return('Return home requested')
        return TriggerResponse(True, 'Returning home')

    def land_srv(self, _):
        if self.state not in ('TAKEOFF', 'EXPLORING', 'RETURNING'):
            return TriggerResponse(False, 'Cannot land now (%s)' % self.state)
        self.setpoint = list(self.position)
        self.start_landing('Landing where it is: the ground below is not checked')
        return TriggerResponse(True, 'Landing')

    def no_pause_srv(self, _):
        return TriggerResponse(False, 'Pause is not available with the learned 3D planner')

    # ------------------------------------------------------------------ planner
    def ask(self):
        """Calls /ari3d/next in a thread (a decision takes up to a second); tick() picks the answer up."""
        self.pending = True

        def call():
            try:
                r = self.next()
                self.answer = (r.success, json.loads(r.message))
            except Exception as e:  # service gone or not up yet
                self.answer = (False, {'reason': 'planner call failed: %r' % (e,)})
        threading.Thread(target=call, daemon=True).start()

    def take_answer(self):
        ok, a = self.answer
        self.answer = None
        self.pending = False
        if not ok:
            self.failures += 1
            self.retry_at = rospy.Time.now() + rospy.Duration(PLAN_RETRY)
            self.event('Planner: %s' % a.get('reason', 'no answer'))
            if self.failures >= PLAN_FAILURES:
                self.start_return('Planner failed %d times in a row' % self.failures)
            return
        self.failures = 0
        if a.get('done'):
            self.start_return('Planner finished (%s)' % a.get('reason', ''))
            return
        self.route = [tuple(q) for q in a['route']]
        self.goal = tuple(a['target'])
        self.plan_pub.publish(String(json.dumps({'goal': self.goal, 'from': self.position, 'path_length': a['route_m'],
                                                 'legs': len(self.route), 'path_time': a['route_m'] / self.speed,
                                                 'plan_time': a['plan_time'], 'fallback': a['fallback']})))
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = 'map'
        for p in [tuple(self.setpoint)] + self.route:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = p
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    # ------------------------------------------------------------------ flight
    def start_return(self, reason):
        self.event(reason + ': retracing the flown path')
        self.route = self.home_route()
        self.setpoint = list(self.position)
        self.set_state('RETURNING')

    def start_landing(self, reason):
        self.event(reason)
        self.route = [(self.setpoint[0], self.setpoint[1], self.home[2])]
        self.set_state('LANDING')

    def home_route(self):
        """Shortest way back over the takeoff spot through remembered positions, so it only uses space already flown."""
        crumbs = self.crumbs + [self.position]
        goal = (self.home[0], self.home[1], self.fly_z)
        goal_index = min(range(len(crumbs)), key=lambda i: dist(crumbs[i], goal))
        cells = {}
        for i, c in enumerate(crumbs):
            cells.setdefault(tuple(int(math.floor(v)) for v in c), []).append(i)

        def neighbours(i):
            cx, cy, cz = (int(math.floor(v)) for v in crumbs[i])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for j in cells.get((cx + dx, cy + dy, cz + dz), ()):
                            d = dist(crumbs[i], crumbs[j])
                            if j != i and (d <= CRUMB_LINK or abs(i - j) == 1):
                                yield j, d

        start = len(crumbs) - 1
        best = {start: 0.0}
        previous = {}
        queue = [(0.0, start)]
        while queue:
            cost, i = heapq.heappop(queue)
            if i == goal_index:
                break
            if cost > best[i]:
                continue
            for j, d in neighbours(i):
                if cost + d < best.get(j, float('inf')):
                    best[j] = cost + d
                    previous[j] = i
                    heapq.heappush(queue, (cost + d, j))
        route = [goal_index]
        while route[-1] in previous:
            route.append(previous[route[-1]])
        route.reverse()
        return [crumbs[i] for i in route] + [goal]

    def follow_route(self, speed, dt):
        """Moves the setpoint along the route. Returns its velocity, or None once the route is done."""
        if not self.route:
            return None
        if self.position is not None and dist(self.setpoint, self.position) > LEASH:
            return [0.0, 0.0, 0.0]
        left = speed * dt
        while self.route:
            d = dist(self.setpoint, self.route[0])
            if d > left:
                direction = [(q - p) / d for p, q in zip(self.setpoint, self.route[0])]
                self.setpoint = [p + left * v for p, v in zip(self.setpoint, direction)]
                return [speed * v for v in direction]
            self.setpoint = list(self.route.pop(0))
            left -= d
        return [0.0, 0.0, 0.0]

    def face(self):
        if self.route and math.hypot(self.route[0][0] - self.setpoint[0], self.route[0][1] - self.setpoint[1]) > FACE_TRAVEL:
            self.yaw_target = math.atan2(self.route[0][1] - self.setpoint[1], self.route[0][0] - self.setpoint[0])

    def send(self, velocity):
        step = max(-self.yaw_rate / RATE, min(self.yaw_rate / RATE, wrap(self.yaw_target - self.yaw)))
        self.yaw = wrap(self.yaw + step)
        target = Target()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = 'map'
        target.type_mask = Target.IGNORE_ACC
        target.position.x, target.position.y, target.position.z = self.setpoint
        target.velocity.x, target.velocity.y, target.velocity.z = velocity
        target.yaw = self.yaw
        self.target_pub.publish(target)

    def tick(self, _):
        if self.state == 'IDLE':
            return
        dt = 1.0 / RATE
        velocity = [0.0, 0.0, 0.0]
        if self.state == 'TAKEOFF':
            velocity = self.follow_route(0.5, dt)
            if not self.route and dist(self.position, self.start_node) < ARRIVED:
                self.event('At the start node: following the learned planner')
                self.explore_since = rospy.Time.now()
                self.set_state('EXPLORING')
        elif self.state == 'EXPLORING':
            if self.answer is not None:
                self.take_answer()
            if self.state != 'EXPLORING':
                pass
            elif (rospy.Time.now() - self.explore_since).to_sec() > self.max_seconds:
                self.start_return('Time limit %.0f s' % self.max_seconds)
            elif self.route:
                self.face()
                velocity = self.follow_route(self.speed, dt)
            elif not self.pending and rospy.Time.now() >= self.retry_at and \
                    (self.goal is None or dist(self.position, self.goal) < ARRIVED):
                self.ask()
        elif self.state == 'RETURNING':
            self.face()
            velocity = self.follow_route(self.speed, dt)
            if not self.route:
                self.start_landing('Over the takeoff spot: landing')
        elif self.state == 'LANDING':
            velocity = self.follow_route(0.3, dt)
            if not self.route:
                self.land_pub.publish(Empty())
                self.event('Landed')
                self.set_state('LANDED')
        self.send(velocity or [0.0, 0.0, 0.0])


if __name__ == '__main__':
    rospy.init_node('ari3d_mission')
    Mission()
    rospy.spin()
