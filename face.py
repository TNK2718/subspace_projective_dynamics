'''https://github.com/TanaTanoi/lets-get-physical-simluation'''

class Face:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def contains_point_ids(self, i, j):
        v_id = self.vertex_ids()
        return i in v_id and j in v_id

    def other_point(self, i, j):
        v_id = set(self.vertex_ids())
        v_id.remove(i)
        v_id.remove(j)
        return v_id.pop()

    def other_points(self, i):
        v_ids = set(self.vertex_ids())
        v_ids.remove(i)
        return list(v_ids)

    def vertex_ids(self):
        return [self.v1, self.v2, self.v3]

    def off_string(self):
        return "3 " + str(self.v1) + " " + str(self.v2) + " " + str(self.v3)

    def center_of_triangle(self, verts):
        return (verts[self.v1] + verts[self.v2] + verts[self.v3]) / 3.0