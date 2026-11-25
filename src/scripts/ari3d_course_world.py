#!/usr/bin/env python3
"""Gazebo world and spawn pose for an evaluation course (voxel_world's make_courses.py folder).

The course's world.sdf includes its room as model://course_<seed>, which Gazebo finds only through a model folder of
that name; the course folders are named course_<seed>_<floor|ceiling>. This writes <course>/world_inline.sdf with
model.sdf's model pasted in (the sun still comes from model://sun), so the world loads by path alone.

  ari3d_course_world.py <course dir>            writes world_inline.sdf (if older than model.sdf) and prints its path
  ari3d_course_world.py <course dir> --spawn    prints "x y z" for the drone: under the start, on the floor
"""
import os
import re
import sys

import numpy as np


def main():
    course = os.path.abspath(os.path.expanduser(sys.argv[1]))
    if len(sys.argv) > 2 and sys.argv[2] == '--spawn':
        z = np.load(os.path.join(course, 'course.npz'))
        s = z['start']
        print('%.3f %.3f %.3f' % (s[0], s[1], 0.1))
        return
    model = os.path.join(course, 'model.sdf')
    out = os.path.join(course, 'world_inline.sdf')
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(model):
        text = open(model).read()
        body = re.search(r'<model\b.*</model>', text, re.S).group(0)
        name = re.search(r'<world name="([^"]+)"', open(os.path.join(course, 'world.sdf')).read())
        world = (name.group(1) if name else 'course')
        with open(out, 'w') as f:
            f.write('<?xml version="1.0"?>\n<sdf version="1.6">\n  <world name="%s">\n'
                    '    <include><uri>model://sun</uri></include>\n%s\n  </world>\n</sdf>\n' % (world, body))
    sys.stdout.write(out)


if __name__ == '__main__':
    main()
