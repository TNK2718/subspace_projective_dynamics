import math
import numpy as np


class Potential:
    def __init__(self, number_of_verts, verts, face, weight):
        self.number_of_verts = number_of_verts
        self.face = face
        self.weight = weight
        # self.A = None

    def calculateRHS(self, verts, b, mass):
        raise Exception

    def A_matrix(self):
        raise Exception

    def calculate_triangle_global_matrix(self, mat):
        raise Exception


class ARAPpotential(Potential):
    def __init__(self, number_of_verts, verts, face, weight, s_max, s_min):
        super().__init__(number_of_verts, verts, face, weight)
        self.s_max = s_max
        self.s_min = s_min

        # TODO: calculate inverse of D_m
        points = face.vertex_ids()
        v1 = verts[points[0]]
        v2 = verts[points[1]]
        v3 = verts[points[2]]

        P_m = np.zeros((3, 3))
        edge1 = v3 - v1
        edge2 = v2 - v1
        P_m[:, 0] = (edge1) / np.linalg.norm(edge1)
        P_m[:, 1] = edge2 - edge2.dot(P_m[:, 0]) * P_m[:, 0]
        P_m[:, 1] /= np.linalg.norm(P_m[:,1])

        dm = P_m.T * np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T
        
        self.area = np.linalg.det(dm) / 2.0
        print(self.area)
        self.dm_I = np.linalg.pinv(dm)
        self.A = self.A_matrix()

    def calculateRHS(self, verts, b, mass):
        # TODO: calculate D_s
        points = self.face.vertex_ids()
        v1 = verts[points[0]]
        v2 = verts[points[1]]
        v3 = verts[points[2]]

        P_s = np.zeros((3, 3))
        edge1 = v3 - v1
        edge2 = v2 - v1
        P_s[:, 0] = (edge1) / np.linalg.norm(edge1)
        P_s[:, 1] = edge2 - edge2.dot(P_s[:, 0]) * P_s[:, 0]
        P_s[:, 1] /= np.linalg.norm(P_s[:,1])

        ds = P_s.T * np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T
        combined = ds.dot(self.dm_I)
        projection = (P_s * self.clamped_svd_for_matrix(combined)).flatten()
        projection = self.A.T.dot(projection.T) * self.weight * math.sqrt(abs(self.area))
        for i in range(9):
            b[3 * points[i // 3] + i % 3] += projection[i]

    def A_matrix(self):
        rslt = np.zeros((9, 9))  # TODO
        for i in range(2):
            for j in range(3):
                rslt[3 * i + j, 3 * 0 + j] = - \
                    (self.dm_I[0, i] + self.dm_I[1, i]) * \
                    math.sqrt(abs(self.area))
                rslt[3 * i + j, 3 * 1 + j] = self.dm_I[0, i] * \
                    math.sqrt(abs(self.area))
                rslt[3 * i + j, 3 * 2 + j] = self.dm_I[1, i] * \
                    math.sqrt(abs(self.area))
        return rslt

    def calculate_triangle_global_matrix(self, mat):
        A_T_A = self.A.T * self.A
        points = self.face.vertex_ids()
        for i in range(9):
            for j in range(9):
                mat[3 * points[i // 3] + i % 3, 3 * points[j // 3] + j %
                    3] += A_T_A[i, j] * self.weight

    def clamped_svd_for_matrix(self, matrix):
        u, s, v_t = np.linalg.svd(matrix)
        s = np.diag(np.clip(s, self.s_min, self.s_max))
        # return np.around(u.dot(s).dot(v_t), 11)
        return u.dot(s).dot(v_t)
