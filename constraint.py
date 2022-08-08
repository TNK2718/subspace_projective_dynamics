import math
import numpy as np


class Constraint:
    def __init__(self, number_of_verts, verts, v_ids, weight):
        self.number_of_verts = number_of_verts
        self.v_ids = v_ids
        self.weight = weight

    def calculateRHS(self, verts, b, mass):
        raise Exception

    def A_matrix(self):
        raise Exception

    def calculate_constraint_global_matrix(self, mat):
        raise Exception


class FixConstraint(Constraint):
    def __init__(self, number_of_verts, verts, v_id, weight):
        super().__init__(number_of_verts, verts, v_id, weight)
        self.A = self.A_matrix()
        self.ini_pos = verts[v_id, :].copy()

    def calculateRHS(self, verts, b, mass):
        projection = self.ini_pos * self.weight  # I^TIp = p
        for i in range(3):
            b[3 * self.v_ids + i] += projection[i]

    def A_matrix(self):
        return np.identity(3)

    def calculate_constraint_global_matrix(self, mat):
        A_T_A = np.identity(3)
        for i in range(3):
            mat[3 * self.v_ids + i, 3 * self.v_ids + i] += 1.0 * self.weight