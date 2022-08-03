import pd_model
import face
import constraint

import numpy.linalg as linalg
import numpy as np

'''For geometry initialization.
Building plane, loading obj file(TODO), etc.

https://github.com/TanaTanoi/lets-get-physical-simluation
'''


def generate_plane(width, height, MAX_WIDTH_SIZE=0.5, MAX_HEIGHT_SIZE=0.3):

    n = width * height
    width_gap = MAX_WIDTH_SIZE / width
    height_gap = -MAX_HEIGHT_SIZE / height
    fix_weight = 1.0

    verts = np.zeros((n, 3))
    faces = []
    constraints = []
    fixed_points = []
    uvs = np.zeros((n, 2))
    for x in range(width):
        for y in range(height):
            verts[x + (y * width)] = np.array((x * width_gap -
                                               MAX_WIDTH_SIZE / 2, y * height_gap - MAX_HEIGHT_SIZE / 2, 0))
            uvs[x + (y * width)] = np.array(((x % width) /
                                             width, 1 - (y % height) / height))

    for v_id in range(n):
        # points before the bottom line
        if v_id % width < width - 1 and v_id < n - width:
            v_1 = v_id
            v_2 = v_id + width
            v_3 = v_id + 1
            add_face(v_1, v_2, v_3, faces)
            # add_spring_constraint_set(
            #     verts, v_1, v_2, v_3, constraints)

        #
        if v_id % width > 0 and v_id < n - width:
            v_1 = v_id + width
            v_2 = v_id
            v_3 = v_id + width - 1
            add_face(v_1, v_2, v_3, faces)

    # fix top and bottom left corners
    # add_fix_constraint(n, verts, 0, fix_weight, constraints)
    bottom_left = width * (height - 1)
    # add_fix_constraint(
    #     n, verts, bottom_left, fix_weight, constraints)

    fixed_points.append(0)
    fixed_points.append(bottom_left)
    return pd_model.PDModel(verts, faces, uvs, constraints=constraints)


def add_face(v_1, v_2, v_3, faces):
    faces.append(face.Face(v_1, v_2, v_3))


def add_fix_constraint(number_of_verts, verts, v_id, weight, constraints):
    constraints.append(constraint.FixConstraint(
        number_of_verts, verts, v_id, weight))
