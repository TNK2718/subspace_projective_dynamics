import time
from matplotlib.cbook import flatten
import numpy as np
import numpy.linalg as linalg
import face
import constraint
import potential
import math

np.set_printoptions(linewidth=2000)


class PDModel:
    def __init__(self, verts, faces, uvs=[], constraints=[], dyn_forces=[], fixed_points=[]):
        '''Geometry'''
        self.n = len(verts)
        self.verts = verts
        self.rendering_verts = np.copy(verts)
        self.faces = faces
        self.rendering_faces = np.zeros((len(self.faces), 3), dtype=int)
        for i in range(len(self.faces)):
            self.rendering_faces[i, 0] = self.faces[i].v1
            self.rendering_faces[i, 1] = self.faces[i].v2
            self.rendering_faces[i, 2] = self.faces[i].v3
        self.verts_to_tri = []
        for i in range(self.n):
            in_faces = []
            for face in self.faces:
                if i in face.vertex_ids():
                    in_faces.append(face)
            self.verts_to_tri.append(in_faces)
        self.uvs = uvs
        self.fixed_points = fixed_points
        self.mass_matrix = np.identity(3 * self.n)  # TODO
        self.mass_matrix /= (len(self.faces))
        self.inv_mass_matrix = np.identity(3 * self.n)  # TODO

        '''Solver option'''
        self.stepsize = 0.3
        self.drag = 1.00
        self.max_iter = 10
        self.eps_n = 0.01  # epsilon for local-global loop(nonlinear solver)

        '''Constraints'''
        # Inner Potential
        self.potential_weight = 1.0 * self.inv_mass_matrix[0, 0] # TODO
        self.potentials = []
        for face in self.faces:
            self.potentials.append(potential.ARAPpotential(
                self.n, self.verts, face, self.potential_weight, 1.0, 0.0))
        # Constraint
        self.constraints = constraints

        '''Variables'''
        self.count = 0
        self.position = verts.flatten()
        self.ini_position = np.copy(self.position)
        self.velocities = np.zeros((3 * self.n))
        self.global_matrix = self.calculate_global_matrix()
        self.inv_global_matrix = np.linalg.inv(self.global_matrix)

        self.dyn_forces = dyn_forces  # Dynamics external forces
        # Static external forces
        self.stat_forces = np.zeros(((3 * self.n)))
        gravity = np.zeros(((self.n, 3)))  # gravity
        gravity[:, 2] = -9.8 * self.mass_matrix[0, 0]  # TODO
        self.stat_forces += gravity.flatten()

        # self.wind_magnitude = 5

    def simulate(self):
        ''' Forces'''
        forces = self.stat_forces
        for f in self.dyn_forces:
            f.add_force()

        '''Inertia'''
        accel = (self.stepsize * self.stepsize) * \
            self.inv_mass_matrix.dot(forces)
        inertia = self.velocities * self.stepsize
        s_0 = self.position + inertia + accel

        '''Local global loop'''
        q_1 = np.copy(s_0)
        b = np.zeros((3 * self.n))
        M = self.mass_matrix / (self.stepsize * self.stepsize)
        b_0 = M.dot(s_0)
        b = np.copy(b_0)

        for iter in range(self.max_iter):
            q_0 = np.copy(q_1)
            b = np.copy(b_0)

            '''Local solve'''
            # Triangle potential
            for potential in self.potentials:
                points = potential.face.vertex_ids()
                avg_mass = 0.0
                for i in range(3):
                    avg_mass += self.mass_matrix[3 * points[i], 3 * points[i]]
                avg_mass /= 3.0
                potential.calculateRHS(self.rendering_verts, b, avg_mass)

            # Constraints
            for con in self.constraints:
                # TODO
                con.calculateRHS(self.rendering_verts, b, 1.0)

            '''Global solve'''
            # q_1 = np.linalg.solve(self.global_matrix, b.flatten())
            q_1 = self.inv_global_matrix.dot(b.flatten())

            for point in self.fixed_points:
                for i in range(3):
                    q_1[3 * point + i] = self.ini_position[3 * point + i]

            self.rendering_verts = q_1.reshape((self.n, 3))

            # break
            diff = np.linalg.norm((q_1 - q_0), ord=2)
            if diff < self.eps_n:
                break

        self.velocities = ((q_1 - self.position)) / self.stepsize
        self.position = np.copy(q_1)
        self.rendering_verts = q_1.reshape((self.n, 3))

    def calculate_global_matrix(self):
        rslt = np.copy(self.mass_matrix)
        rslt /= (self.stepsize * self.stepsize)

        for potential in self.potentials:
            potential.calculate_triangle_global_matrix(rslt)

        for constraint in self.constraints:
            constraint.calculate_constraint_global_matrix(rslt)

        return rslt
