from email.mime import base
import time
from matplotlib.cbook import flatten
import numpy as np
import numpy.linalg as linalg
import face
import constraint
import potential
import math

np.set_printoptions(linewidth=2000)

'''Semi-Reduced Projective Dynamics'''


class SubPDModel:
    def __init__(self, verts, faces, base_mat, center, uvs=[], constraints=[], dyn_forces=[], fixed_points=[]):
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
        self.uvs = uvs
        self.fixed_points = fixed_points
        self.mass_matrix = np.identity(3 * self.n)  # TODO
        self.mass_matrix /= (len(self.faces))
        self.inv_mass_matrix = np.identity(3 * self.n)  # TODO

        '''Solver option'''
        self.stepsize = 0.1
        self.drag = 1.00
        self.max_iter = 10
        self.eps_n = 0.001  # epsilon for local-global loop(nonlinear solver)

        '''Constraints'''
        # Inner Potential
        self.potential_weight = 2.0
        self.potentials = []
        for face in self.faces:
            self.potentials.append(potential.ARAPpotential(
                self.n, self.rendering_verts, face, self.potential_weight, 1.0, 0.0))
        # Constraint
        self.constraints = constraints

        '''Variables: fullspace'''
        self.count = 0
        self.position = verts.flatten()
        self.ini_position = np.copy(self.position)
        self.velocities = np.zeros((3 * self.n))
        self.global_matrix = self.calculate_global_matrix()
        self.inv_global_matrix = np.linalg.inv(self.global_matrix)

        self.dyn_forces = dyn_forces  # Dynamic external forces
        # Static external forces
        self.stat_forces = np.zeros(((3 * self.n)))
        gravity = np.zeros(((self.n, 3)))  # gravity
        gravity[:, 2] = -9.8 * self.mass_matrix[0, 0]  # TODO
        self.stat_forces += gravity.flatten()

        '''Variables: subspace'''
        self.U = base_mat
        self.center = center
        self.sub_position = self.to_subspace(self.position)
        self.sub_velocities = self.U.T.dot(self.velocities)
        self.sub_global_mat = self.U.T @ self.global_matrix @ self.U
        self.sub_inv_global_mat = np.linalg.inv(self.sub_global_mat)
        self.sub_M = self.U.T @ self.mass_matrix @ self.U / \
            (self.stepsize * self.stepsize)
        self.sub_inv_mass_mat = self.U.T @ self.inv_mass_matrix @ self.U
        self.sub_fict_force = self.U.T @ (self.sub_M - self.global_matrix) @ self.center

    def simulate(self):
        '''
        Forces
        TODO: Perform solely in subspace
        Dynamic forces and collisions are evaluated in fullspace.
        '''
        forces = self.stat_forces
        for f in self.dyn_forces:
            f.add_force()
        sub_forces = self.U.T @ forces

        '''Inertia'''
        sub_accel = (self.stepsize * self.stepsize) * self.sub_inv_mass_mat @ sub_forces
        sub_inertia = self.sub_velocities * self.stepsize
        sub_s_0 = self.sub_position + sub_inertia + sub_accel

        # accel = (self.stepsize * self.stepsize) * \
        #     self.inv_mass_matrix.dot(forces)
        # inertia = self.velocities * self.stepsize
        # s_0 = self.position + inertia + accel

        '''Local global loop'''
        sub_q_1 = np.copy(sub_s_0)
        sub_b_0 = self.sub_M @ sub_s_0 + self.sub_fict_force
        sub_b = np.copy(sub_b_0)
        
        # q_1 = np.copy(s_0)
        # b = np.zeros((3 * self.n))
        # M = self.mass_matrix / (self.stepsize * self.stepsize)
        # b_0 = M.dot(s_0)
        # b = np.copy(b_0)

        for iter in range(self.max_iter):
            sub_q_0 = np.copy(sub_q_1)
            sub_b = np.copy(sub_b_0)

            # q_0 = np.copy(q_1)
            # b = np.copy(b_0)

            '''
            Local solve in fullspace
            
            TODO: Perform local solve solely in subspace
            HRPD[Brandt et al.2018] enables to perform local solve in subspace, leveraging sampled fullspace infomation.
            However, this sampling-based method can not be applied to cloth simulation due to its high-frequency deformations.
            '''
            vec = np.zeros((3 * self.n))
            # Triangle potential
            for potential in self.potentials:
                points = potential.face.vertex_ids()
                avg_mass = 0.0
                for i in range(3):
                    avg_mass += self.mass_matrix[3 * points[i], 3 * points[i]]
                avg_mass /= 3.0
                potential.calculateRHS(self.rendering_verts, vec, avg_mass)

            # Constraints
            for con in self.constraints:
                # TODO
                con.calculateRHS(self.rendering_verts, vec, 0.0)
            
            sub_b += self.U.T @ vec

            '''Global solve in subspace'''
            sub_q_1 = self.inv_global_matrix.dot(sub_b.flatten())

            self.position = self.to_fullspace(sub_q_1)
            self.rendering_verts = self.position.copy().reshape((self.n, 3))

            # break
            diff = np.linalg.norm((sub_q_1 - sub_q_0), ord=2)
            if diff < self.eps_n:
                break

        self.sub_velocities = (sub_q_1 - self.sub_position) / self.stepsize
        self.sub_position = sub_q_1

        self.velocities = self.U @ self.sub_velocities

    def calculate_global_matrix(self):
        rslt = np.copy(self.mass_matrix)
        rslt /= (self.stepsize * self.stepsize)

        for potential in self.potentials:
            potential.calculate_triangle_global_matrix(rslt)

        for constraint in self.constraints:
            constraint.calculate_constraint_global_matrix(rslt)

        return rslt

    def to_subspace(self, full_pos):
        return self.U.T @ (full_pos - self.center)

    def to_fullspace(self, sub_pos):
        return self.U @ sub_pos + self.center