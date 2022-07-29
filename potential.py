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
    
    def S_matrix(self):
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
        dm = np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T
        self.area = np.linalg.det(dm) / 2.0
        self.dm_I = np.linalg.pinv(dm)
        self.A = self.A_matrix()

    def calculateRHS(self, verts, b, mass):
        # TODO: calculate D_s
        points = self.face.vertex_ids()
        v1 = verts[points[0]]
        v2 = verts[points[1]]
        v3 = verts[points[2]]
        ds = np.matrix((v3 - v1, v2 - v1, (0, 0, 0))).T
        combined = ds.dot(self.dm_I)
        projection = self.clamped_svd_for_matrix(combined).flatten()
        projection = self.A.T.dot(projection) / mass
        for i in range(9):
            b[3 * points[i / 3] + i % 3] += projection[i]


    def A_matrix(self):
        rslt = np.zeros((9, 9)) # TODO
        for i in range(2):
            for j in range(3):
                rslt[3 * i + j, 3 * 0 + j] = -(self.dm_I[0, i] + self.dm_I[1, i]) * math.sqrt(abs(self.area) * self.weight)
                rslt[3 * i + j, 3 * 1 + j] = self.dm_I[0, i] * math.sqrt(abs(self.area) * self.weight)
                rslt[3 * i + j, 3 * 2 + j] = self.dm_I[1, i] * math.sqrt(abs(self.area) * self.weight)
        return rslt
    
    def S_matrix(self):
        rslt = np.zeros((9, 3 * self.number_of_verts))
        points = self.face.vertex_ids()
        for i in range(9):
            rslt[i, 3 * points[i / 3] + i % 3] = 1.0
        return rslt

    def clamped_svd_for_matrix(self, matrix):
        u, s, v_t = np.linalg.svd(matrix)
        s = np.diag(np.clip(s, 0, 1.0))
        # return np.around(u.dot(s).dot(v_t), 11)
        return u.dot(s).dot(v_t)    