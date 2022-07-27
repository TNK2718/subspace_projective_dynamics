import numpy as np
import numpy.linalg as linalg
import face
import constraint
import math

np.set_printoptions(linewidth=2000)

default_flag_type = "spring"


class PDModel:
    def __init__(self, verts, faces, uvs=[], constraints=[], dyn_forces=[]):
        '''Geometry'''
        self.n = len(verts)
        self.verts = verts
        self.rendering_verts = np.copy(verts)
        self.faces = faces
        self.verts_to_tri = []
        for i in range(self.n):
            in_faces = []
            for face in self.faces:
                if i in face.vertex_ids():
                    in_faces.append(face)
            self.verts_to_tri.append(in_faces)
        self.uvs = uvs

        '''Solver option'''
        self.stepsize = 0.3
        self.drag = 1.00
        self.max_iter = 30
        self.eps_n = 0.001 # epsilon for local-global loop(nonlinear solver)

        '''Variables'''
        self.count = 0
        self.position = verts.flatten()
        self.velocities = np.zeros((3 * self.n))
        self.mass_matrix = np.identity(3 * self.n)
        self.mass_matrix /= (len(self.faces))
        self.global_matrix = self.calculate_triangle_global_matrix()
        self.dyn_forces = dyn_forces  # Dynamics external forces
        # Static external forces
        self.stat_forces = np.zeros(((3 * self.n)))
        gravity = np.zeros(((self.n, 3)))  # gravity
        gravity[:, 1] = -9.8
        self.stat_forces += gravity.flatten()

        # self.wind_magnitude = 5

        '''Constraints'''
        self.constraints = constraints
        self.fixed_points = []
        for con in self.constraints:
            if con.type() == "FIXED":
                self.fixed_points.append(con)

    def simulate(self):
        ''' Forces'''
        forces = self.stat_forces
        for f in self.dyn_forces:
            f.add_force()

        '''Inertia'''
        accel = (self.stepsize * self.stepsize) * \
            linalg.inv(self.mass_matrix).dot(forces)
        inertia = self.velocities * self.stepsize
        s_0 = self.position + inertia + accel
        
        '''Local global loop'''
        q_1 = np.copy(s_0)
        b_array = np.zeros((self.n + len(self.fixed_points), 3))
        M = self.mass_matrix / (self.stepsize * self.stepsize)
        b_array[:self.n] = M.dot(s_0)
        flag = False

        for _ in range(self.max_iter):
            q_0 = np.copy(q_1)
            '''Local solve'''
            # Triangle potential
            for face in self.faces:
                f_verts = face.vertex_ids()
                for i in range(3):
                    v1 = f_verts[i]
                    v2 = f_verts[(i + 1) % 3]
                    T = self.potential_for_triangle(face, q_1, v2)
                    edge = self.verts[v2] - self.verts[v1]
                    g = T.dot(edge)
                    b_array[v1] = b_array[v1] - g
                    b_array[v2] = b_array[v2] + g

            # Constraints
            for con_i in range(len(self.fixed_points)):
                con = self.fixed_points[con_i]
                b_array[-(con_i + 1)] = self.verts[con.vert_a]

            '''Global solve'''
            q_1 = np.linalg.solve(self.global_matrix, b_array.flatten())
            # Don't grab the unwanted fixed points
            q_1 = q_1[:-3 * len(self.fixed_points)]

            # break
            diff = np.linalg.norm((q_1 - q_0), ord=2)
            if diff < self.eps_n:
                break

        self.position = np.copy(q_1)
        self.velocities = ((q_1 - self.rendering_verts)) / self.stepsize
        self.rendering_verts = q_1.reshape((self.n ,3))

    def center(self):
        middle_point = np.array((0., 0., 0.))
        for vert in self.verts:
            middle_point += vert
        middle_point = middle_point / float(self.n)
        middle_point *= -1
        for vert_id in range(self.n):
            self.verts[vert_id] += middle_point

    def calculate_b_for_triangle(self, i, s_0):
        b = np.zeros((1, 3))
        for face in self.verts_to_tri[i]:
            T = self.potential_for_triangle(face, s_0, i)
            for o_v in face.other_points(i):
                b += T.dot(self.verts[i] - self.verts[o_v])
        return b

    def potential_for_triangle(self, face, prime_verts, point):
        other_points = face.other_points(point)
        v1 = self.verts[point]
        v2 = self.verts[other_points[0]]
        v3 = self.verts[other_points[1]]
        x_g = np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T

        v1 = prime_verts[point]
        v2 = prime_verts[other_points[0]]
        v3 = prime_verts[other_points[1]]
        x_f = np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T

        combined = x_f.dot(np.linalg.pinv(x_g))
        return self.clamped_svd_for_matrix(combined)

    def clamped_svd_for_matrix(self, matrix):
        u, s, v_t = np.linalg.svd(matrix)
        s = np.diag(np.clip(s, 0, 1.0))
        # return np.around(u.dot(s).dot(v_t), 11)
        return u.dot(s).dot(v_t)

    def calculate_triangle_global_matrix(self):
        fixed_point_num = len(self.fixed_points)
        M = np.zeros((self.n + fixed_point_num, self.n + fixed_point_num))
        M[:self.n, :self.n] = self.mass_matrix
        M /= (self.stepsize * self.stepsize)

        weights = np.zeros(
            (self.n + fixed_point_num, self.n + fixed_point_num))
        weight_sum = np.zeros(
            (self.n + fixed_point_num, self.n + fixed_point_num))
        for face in self.faces:
            verts = face.vertex_ids()
            for k in range(3):
                v_1 = verts[k]
                v_2 = verts[(k + 1) % 3]
                weights[v_1, v_2] += 1
                weights[v_2, v_1] += 1
                weight_sum[v_1, v_1] += 1
                weight_sum[v_2, v_2] += 1

        x = weight_sum - weights
        for i in range(fixed_point_num):
            con = self.fixed_points[i]
            x[con.vert_a, -(i + 1)] = 1
            x[-(i + 1), con.vert_a] = 1
            M[-(i + 1), -(i + 1)] = 0
        return x + M
