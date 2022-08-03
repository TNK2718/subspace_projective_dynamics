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
    fix_weight = 100

    verts = np.zeros((n, 3))
    faces = []
    constraints = []
    uvs = np.zeros((n, 2))
    for x in range(width):
        for y in range(height):
            verts[x + (y * width)] = np.array((x *
                                               width_gap, y * height_gap, 0))
            uvs[x + (y * width)] = np.array(((x % width) /
                                             width, 1 - (y % height) / height))

    # for v_id in range(n):
    #     # if its a thing on the end
    #     if v_id % width == width - 1:
    #         if v_id < n - 1:
    #             add_spring_constraint(
    #                 verts, v_id, v_id + width, constraints)
    #         continue
    #     # points before the bottom line
    #     if v_id < n - width:
    #         v_1 = v_id
    #         v_2 = v_id + width
    #         v_3 = v_id + 1
    #         add_face(v_1, v_2, v_3, faces)
    #         add_spring_constraint_set(
    #             verts, v_1, v_2, v_3, constraints)
    #     # points after the first line
    #     if v_id >= width:
    #         v_1 = v_id
    #         v_2 = v_id + 1
    #         v_3 = v_id - (width - 1)
    #         add_face(v_1, v_2, v_3, faces)
    #     # the lines along the bottom
    #     if v_id >= n - width and v_id < n:
    #         add_spring_constraint(
    #             verts, v_id, v_id + 1, constraints)

    # fix top and bottom left corners
    add_fixed_constraint(n, verts, 0, fix_weight, constraints)
    bottom_left = width * (height - 1)
    add_fixed_constraint(
        n, verts, bottom_left, fix_weight, constraints)
    return pd_model.PDModel(verts, faces, uvs, constraints=constraints)


def add_face(v_1, v_2, v_3, faces):
    faces.append(face.Face(v_1, v_2, v_3))


def add_fixed_constraint(number_of_verts, verts, v_id, weight, constraints):
    constraints.append(constraint.FixConstraint(
        number_of_verts, verts, v_id, weight))
