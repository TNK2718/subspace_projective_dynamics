import numpy as np
import numpy.linalg as linalg
import face
import constraint
import potential
import math

np.set_printoptions(linewidth=2000)

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
        self.eps_n = 0.001  # epsilon for local-global loop(nonlinear solver)

        '''Variables'''
        self.count = 0
        self.position = verts.flatten()
        self.velocities = np.zeros((3 * self.n))
        self.mass_matrix = np.identity(3 * self.n)  # TODO
        self.mass_matrix /= (len(self.faces))
        self.inv_mass_matrix = np.identity(3 * self.n) # TODO
        self.global_matrix = self.calculate_triangle_global_matrix()
        self.dyn_forces = dyn_forces  # Dynamics external forces
        # Static external forces
        self.stat_forces = np.zeros(((3 * self.n)))
        gravity = np.zeros(((self.n, 3)))  # gravity
        gravity[:, 1] = -9.8
        self.stat_forces += gravity.flatten()

        # self.wind_magnitude = 5

        '''Constraints'''
        #
        self.potential_weight = 0.01
        self.potentials = []
        for face in self.faces:
            self.potentials.append(potential.ARAPpotential(
                self.n, self.verts, face, self.potential_weight, 1.0, 0.0))

        #
        self.constraints = constraints

    def simulate(self):
        ''' Forces'''
        forces = self.stat_forces
        for f in self.dyn_forces:
            f.add_force()

        '''Inertia'''
        accel = (self.stepsize * self.stepsize) * self.inv_mass_matrix.dot(forces)
        inertia = self.velocities * self.stepsize
        s_0 = self.position + inertia + accel

        '''Local global loop'''
        q_1 = np.copy(s_0)
        b = np.zeros((3 * self.n))
        M = self.mass_matrix / (self.stepsize * self.stepsize)
        b[:self.n] = M.dot(s_0)
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
                    b[v1] = b[v1] - g
                    b[v2] = b[v2] + g

            # Constraints
            for con_i in range(len(self.fixed_points)):
                con = self.fixed_points[con_i]
                b[-(con_i + 1)] = self.verts[con.vert_a]

            '''Global solve'''
            q_1 = np.linalg.solve(self.global_matrix, b.flatten())
            # Don't grab the unwanted fixed points
            q_1 = q_1[:-3 * len(self.fixed_points)]

            # break
            diff = np.linalg.norm((q_1 - q_0), ord=2)
            if diff < self.eps_n:
                break

        self.position = np.copy(q_1)
        self.velocities = ((q_1 - self.rendering_verts)) / self.stepsize
        self.rendering_verts = q_1.reshape((self.n, 3))

    def calculate_triangle_global_matrix(self):
        rslt = np.copy(self.mass_matrix)
        rslt /= (self.stepsize * self.stepsize)

        for potential in self.potentials:
            points = potential.face.vertex_ids()
            avg_inv_mass = 0.0
            for i in range(3):
                avg_inv_mass += self.inv_mass_matrix[points[i], points[i]]
            avg_inv_mass /= 3.0
            S = potential.S_matrix()
            rslt += avg_inv_mass * S.T @ potential.A.T @ potential.A @ S 

        # for constraint in self.constraints:
        #     points = constraint.face.vertex_ids()
        #     avg_inv_mass = 0.0
        #     for i in range(3):
        #         avg_inv_mass += self.inv_mass_matrix[points[i], points[i]]
        #     avg_inv_mass /= 3.0
        #     S = constraint.S_matrix()
        #     rslt += avg_inv_mass * S.T * constraint.A.T * constraint.A * S 

        return rslt        
