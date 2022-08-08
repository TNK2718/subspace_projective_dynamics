from termios import VDISCARD
import pd_model
import face
import constraint

import numpy.linalg as linalg
import numpy as np
import math

'''For geometry initialization.
Building plane, loading obj file(TODO), etc.
Returns PD solver model
'''

def generate_plane(width, height, MAX_WIDTH_SIZE=0.5, MAX_HEIGHT_SIZE=0.3, subspace=False):

    n = width * height
    width_gap = MAX_WIDTH_SIZE / width
    height_gap = -MAX_HEIGHT_SIZE / height
    fix_weight = 100000.0

    verts = np.zeros((n, 3))
    faces = []
    constraints = []
    fixed_points = []
    uvs = np.zeros((n, 2))
    for x in range(width):
        for y in range(height):
            verts[x + (y * width)] = np.array((x * width_gap -
                                               MAX_WIDTH_SIZE / 2, y * height_gap + MAX_HEIGHT_SIZE / 2, 0))
            uvs[x + (y * width)] = np.array(((x % width) /
                                             width, 1 - (y % height) / height))

    # for v_id in range(n):
    #     # points before the bottom line
    #     if v_id % width < width - 1 and v_id < n - width:
    #         v_1 = v_id
    #         v_2 = v_id + width
    #         v_3 = v_id + 1
    #         add_face(v_1, v_2, v_3, faces)
    #         # add_spring_constraint_set(
    #         #     verts, v_1, v_2, v_3, constraints)

    #     #
    #     if v_id % width > 0 and v_id < n - width:
    #         v_1 = v_id + width
    #         v_2 = v_id
    #         v_3 = v_id + width - 1
    #         add_face(v_1, v_2, v_3, faces)

        # # Original ver.
        # if v_id % width == width - 1:
        #     continue
        # # points before the bottom line
        # if v_id < n - width:
        #     v_1 = v_id
        #     v_2 = v_id + width
        #     v_3 = v_id + 1
        #     add_face(v_1, v_2, v_3, faces)
        # # points after the first line
        # if v_id >= width:
        #     v_1 = v_id
        #     v_2 = v_id + 1
        #     v_3 = v_id - (width - 1)
        #     add_face(v_1, v_2, v_3, faces)

    for i in range(height):
        for j in range(width):
            if i < height - 1 and j < width - 1:
                v_1 = width * i + j
                v_2 = width * i + j + 1
                v_3 = width * (i + 1) + j + 1
                add_face(v_1, v_2, v_3, faces)

                v_1 = width * i + j
                v_2 = width * (i + 1) + j
                v_3 = width * (i + 1) + j + 1
                add_face(v_1, v_2, v_3, faces)

    # fix top and bottom left corners
    add_fix_constraint(n, verts, 0, fix_weight, constraints)
    bottom_left = width * (height - 1)
    add_fix_constraint(
        n, verts, bottom_left, fix_weight, constraints)

    fixed_points.append(0)
    fixed_points.append(bottom_left)
    
    # Dynamic force
    dynamic_forces = []
    return (verts, faces, uvs, constraints, dynamic_forces, fixed_points)

# TODO
def generate_iso_plane(width, height, MAX_WIDTH_SIZE=0.5, MAX_HEIGHT_SIZE=0.3):
    n = (2 * width + 1) * height + width + 1
    width_gap = MAX_WIDTH_SIZE / width
    height_gap = -MAX_HEIGHT_SIZE / height
    fix_weight = 100000.0

    verts = np.zeros((n, 3))
    faces = []
    constraints = []
    fixed_points = []
    uvs = np.zeros((n, 2))

    # for x in range(width):
    #     for y in range(height):
    #         verts[x + (y * width)] = np.array((x * width_gap -
    #                                            MAX_WIDTH_SIZE / 2, y * height_gap + MAX_HEIGHT_SIZE / 2, 0))
    #         uvs[x + (y * width)] = np.array(((x % width) /
    #                                          width, 1 - (y % height) / height))

    for v_id in range(n):
        if v_id % (2 * width + 1) >= (width + 1):
            x = math.sqrt(2) * width_gap * (v_id // (2 * width + 1) + 0.5)
            y = math.sqrt(2) * width_gap * (v_id % (2 * width + 1) -
                                            (width + 1)) - math.sqrt(2) * 0.5 * width_gap
            z = 0.0
            verts[v_id] = np.array((x, y, z))
        else:
            x = math.sqrt(2) * width_gap * (v_id // (2 * width + 1))
            y = math.sqrt(2) * width_gap * (v_id %
                                            (2 * width + 1)) - math.sqrt(2) * width_gap
            z = 0.0
            verts[v_id] = np.array((x, y, z))

    for i in range(height):
        for j in range(width):
            v_1 = (2 * width + 1) * i + j
            v_2 = (2 * width + 1) * i + j + 1
            v_3 = (2 * width + 1) * i + j + width + 1
            add_face(v_1, v_2, v_3, faces)

            v_1 = (2 * width + 1) * i + j
            v_2 = (2 * width + 1) * i + j + width + 1
            v_3 = (2 * width + 1) * i + j + 2 * width + 1
            add_face(v_1, v_2, v_3, faces)

            v_1 = (2 * width + 1) * i + j + 1
            v_2 = (2 * width + 1) * i + j + 2 * width + 2
            v_3 = (2 * width + 1) * i + j + width + 1
            add_face(v_1, v_2, v_3, faces)

            v_1 = (2 * width + 1) * i + j + width + 1
            v_2 = (2 * width + 1) * i + j + 2 * width + 2
            v_3 = (2 * width + 1) * i + j + 2 * width + 1
            add_face(v_1, v_2, v_3, faces)

    # fix top and bottom left corners
    add_fix_constraint(n, verts, 0, fix_weight, constraints)
    bottom_left = width * (height - 1)
    add_fix_constraint(
        n, verts, bottom_left, fix_weight, constraints)

    fixed_points.append(0)
    fixed_points.append(bottom_left)

    # Dynamic force
    dynamic_forces = []
    return (verts, faces, uvs, constraints, dynamic_forces, fixed_points)


def add_face(v_1, v_2, v_3, faces):
    faces.append(face.Face(v_1, v_2, v_3))


def add_fix_constraint(number_of_verts, verts, v_id, weight, constraints):
    constraints.append(constraint.FixConstraint(
        number_of_verts, verts, v_id, weight))
