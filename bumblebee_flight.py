from tkinter import E
import pygame
from pygame import draw
import pygame_widgets
from pygame_widgets.button import Button, ButtonArray
import numpy as np
from enum import Enum
from numba import jit
import math
import time


class Color(Enum):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (211, 211, 211)


class Mode(Enum):
    MOVING = 0
    ROTATING = 1


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    FORWARD = 4
    BACKWARDS = 5


class Shading(Enum):
    WIREFRAME = 0
    LAMBERT = 1
    GOURAUD = 2
    PHONG = 3
    PHONG_RAYS = 4


SCREEN_WIDTH = 800#1024
SCREEN_HEIGHT = 600#512

FOV = np.radians(90)

FOCAL_LENGTH = SCREEN_WIDTH / 2 * np.cos(FOV / 2) / np.sin(FOV / 2)

FPS = 60

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class Point3D():
    def __init__(self, x=0, y=0, z=0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __sub__(self, other):
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_arr(self):
        return np.array([self.x, self.y, self.z])

    def rotate_X(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        y = self.y * cosA - self.z * sinA
        z = self.y * sinA + self.z * cosA
        return Point3D(self.x, y, z)

    def rotate_Y(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        z = self.z * cosA - self.x * sinA
        x = self.z * sinA + self.x * cosA
        
        return Point3D(x, self.y, z)

    def rotate_Z(self, angle):
        sinA, cosA = self.__determ_sincos(angle)
        x = self.x * cosA - self.y * sinA
        y = self.x * sinA + self.y * cosA
        return Point3D(x, y, self.z)

    def rotate(self, x, y, z):
        x, y, z = x * np.pi / 180, y * np.pi / 180, z * np.pi / 180
        sin_a, cos_a = np.sin(x), np.cos(x)
        sin_b, cos_b = np.sin(y), np.cos(y)
        sin_g, cos_g = np.sin(z), np.cos(z)

        matrix = np.array([
            [cos_b * cos_g , sin_a * sin_b * cos_g - cos_a * sin_g, cos_a * sin_b * cos_g + sin_a * sin_g],
            [cos_b * sin_g, sin_a * sin_b * sin_g + cos_a * cos_g, cos_a * sin_b * sin_g - sin_a * cos_g],
            [-sin_b, sin_a * cos_b, cos_a * cos_b]
        ])
        x, y, z = np.dot(matrix, np.array([self.x, self.y, self.z]))
        return Point3D(x, y, z)

    def rotate_yx(self, x, y, z):
        x_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        x_matrix[:3, :3] = rotation_matrix([1, 0, 0], x)
        y_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        y_matrix[:3, :3] = rotation_matrix([0, 1, 0], y)

        point = np.array([self.x, self.y, self.z, 1])
        point = np.dot(x_matrix, np.dot(y_matrix, point))
        return Point3D(*point[:3])

    def rotate_xy(self, x, y, z):
        x_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        x_matrix[:3, :3] = rotation_matrix([1, 0, 0], x)
        y_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        y_matrix[:3, :3] = rotation_matrix([0, 1, 0], y)

        point = np.array([self.x, self.y, self.z, 1])
        point = np.dot(y_matrix, np.dot(x_matrix, point))
        return Point3D(*point[:3])

    def project(self):
        if (self.z) > 0:
            factor = FOCAL_LENGTH / (self.z)
            x = (self.x) * factor + SCREEN_WIDTH / 2
            y = (self.y) * factor + SCREEN_HEIGHT / 2
        else:
            x = y = 0
        return Point3D(x, y, self.z)

    def __determ_sincos(self, angle):
        rad = angle * np.pi / 180
        cosA = np.cos(rad)
        sinA = np.sin(rad)

        return sinA, cosA


class PointLight():
    def __init__(self, point, color):
        self.point = point
        self.color = color
    
    def translate_cube(self, x, y, z):
        self.point = Point3D(*(self.point.to_arr() + np.array([x, y, z])))

    def rotate_cube(self, direction, speed):
        pass

    def to_arr(self):
        return np.array([*self.point.to_arr(), *self.color])


@jit(nopython=True, fastmath=True)
def get_c_and_norm(faces, vertices, return_planes=False):
    normals = np.zeros((faces.shape[0], 4 if return_planes else 3))
    centers = np.zeros((faces.shape[0], 3))
    for i, face in enumerate(faces):
        center = np.sum(vertices[face], axis=0) / 4
        p1 = vertices[face[0]]
        p2 = vertices[face[1]]
        p3 = vertices[face[2]]
        v1 = p3 - p1
        v2 = p2 - p1
        normal = -np.cross(v1, v2)

        if return_planes:
            d = np.dot(normal, p1)
            normals[i] = [*normal, d]
        else:
            normals[i] = normal
        centers[i] = center

    return centers, normals

@jit(nopython=True, fastmath=True)
def lambert_lighting(faces, vertices, point_lights, color):
    intensities = np.zeros((len(faces), 3))
    centers, normals = get_c_and_norm(faces, vertices)
    for light in point_lights:
        light_pos = light[:3]
        light_color = light[3:]
        light_vectors = light_pos - centers
        for i in np.arange(len(light_vectors)):
            light_vector = light_vectors[i] / np.linalg.norm(light_vectors[i])
            normal = normals[i] / np.linalg.norm(normals[i])
            intensity_coef = np.maximum(np.dot(light_vector, normal), 0)
            intensities[i] += color * light_color * intensity_coef
    return np.minimum(intensities, 1)

@jit(nopython=True, fastmath=True)
def gouraud_lighting(faces, vertices, vertices_faces, point_lights, color):
    intensities = np.zeros((len(vertices), 3))
    _, faces_normals = get_c_and_norm(faces, vertices)
    vertices_normals = np.zeros((len(vertices), 3))
    for i, faces in enumerate(vertices_faces):
        for v in faces:
            vertices_normals[i] += faces_normals[v]
        vertices_normals[i] /= len(faces)

    for light in point_lights:
        light_pos = light[:3]
        light_color = light[3:]
        light_vectors = light_pos - vertices
        
        for i in np.arange(len(light_vectors)):
            light_vector = light_vectors[i] / np.linalg.norm(light_vectors[i])
            normal = vertices_normals[i] / np.linalg.norm(vertices_normals[i])
            intensity_coef = np.maximum(np.dot(light_vector, normal), 0)
            intensities[i] += color * light_color * intensity_coef

    return np.minimum(intensities, 1)

class Cube():
    def __init__(self, size=1, color=None):
        if color is None:
            color = [1, 1, 1]

        self.vertices = [
            Point3D(-1 * size, 1 * size, -1 * size),
            Point3D(1 * size, 1 * size, -1 * size),
            Point3D(1 * size, -1 * size, -1 * size),
            Point3D(-1 * size, -1 * size, -1 * size),
            Point3D(-1 * size, 1 * size, 1 * size),
            Point3D(1 * size, 1 * size, 1 * size),
            Point3D(1 * size, -1 * size, 1 * size),
            Point3D(-1 * size, -1 * size, 1 * size)
        ]
        self.faces = [(0, 1, 2, 3),
                      (1, 5, 6, 2),
                      (5, 4, 7, 6),
                      (4, 0, 3, 7),
                      (0, 4, 5, 1),
                      (3, 2, 6, 7)]


        self.vertices_faces = []
        for _ in range(len(self.vertices)):
            self.vertices_faces.append([])
        for i, face in enumerate(self.faces):
            for j in face:
                self.vertices_faces[j].append(i)
        
        self.angles = [0, 0, 0]
        self.pos = [0, 0, 0]
        self.lambert_colors = np.zeros((len(self.faces), 3), dtype=int)
        self.gouraud_colors = np.zeros((len(self.vertices), 3), dtype=int)
        self.color = np.array(color)

    def lambert_make_color(self, point_lights):
        if len(point_lights) == 0: return
        t_vertices = self.transform_vertices([0, 0, 0], [0, 0, 0], False, True)
        self.lambert_colors = lambert_lighting(np.array(self.faces), np.array(t_vertices), np.array([pl.to_arr() for pl in point_lights]), self.color) * 255

    def gouraud_make_color(self, point_lights):
        if len(point_lights) == 0: return
        t_vertices = self.transform_vertices([0, 0, 0], [0, 0, 0], False, True)
        self.gouraud_colors = gouraud_lighting(np.array(self.faces), np.array(t_vertices), np.array(self.vertices_faces), np.array([pl.to_arr() for pl in point_lights]), self.color) * 255

    def draw_cube(self, camera_pos, camera_angles, point_lights, shading=None, triangles=True):
        t_vertices = self.transform_vertices(camera_pos, camera_angles, True)

        avg_Z = self.calculate_avg_z(t_vertices)
        polygons = []

        _, normals = self.get_c_and_norm(t_vertices, True)
        vis = np.dot([0, 0, -1, 0], normals.T)
        
        for z_val in sorted(avg_Z, key=lambda x: x[1], reverse=True):
            if vis[z_val[0]] <= 0:
                continue
            if (z_val[1]) < 3:
                continue
            f_index = z_val[0]
            f = self.faces[f_index]

            if triangles:
                point_list = np.array([
                    (t_vertices[f[0]].x, t_vertices[f[0]].y, t_vertices[f[0]].z, f[0]),
                    (t_vertices[f[1]].x, t_vertices[f[1]].y, t_vertices[f[1]].z, f[1]),
                    (t_vertices[f[2]].x, t_vertices[f[2]].y, t_vertices[f[2]].z, f[2]),
                ])

                polygons.append((point_list, f_index))

                point_list = np.array([
                    (t_vertices[f[0]].x, t_vertices[f[0]].y, t_vertices[f[0]].z, f[0]),
                    (t_vertices[f[2]].x, t_vertices[f[2]].y, t_vertices[f[2]].z, f[2]),
                    (t_vertices[f[3]].x, t_vertices[f[3]].y, t_vertices[f[3]].z, f[3]),
                ])
                polygons.append((point_list, f_index))
            else:

                point_list = np.array([
                    (t_vertices[f[0]].x, t_vertices[f[0]].y, t_vertices[f[0]].z, f[0]),
                    (t_vertices[f[1]].x, t_vertices[f[1]].y, t_vertices[f[1]].z, f[1]),
                    (t_vertices[f[2]].x, t_vertices[f[2]].y, t_vertices[f[2]].z, f[2]),
                    (t_vertices[f[3]].x, t_vertices[f[3]].y, t_vertices[f[3]].z, f[3]),
                ])
                polygons.append((point_list, f_index))

        if len(polygons) > 0:
            if shading == Shading.LAMBERT:
                self.lambert_make_color(point_lights)
            elif shading == Shading.GOURAUD:
                self.gouraud_make_color(point_lights)
        
        return polygons

    def translate_cube(self, x, y, z):
        self.pos[0] += x
        self.pos[1] += y
        self.pos[2] += z

    def rotate_cube(self, direction, speed = 1):
        if direction == Direction.UP:
            self.angles[0] += 2 * speed
        elif direction == Direction.DOWN:
            self.angles[0] -= 2 * speed
        elif direction == Direction.LEFT:
            self.angles[1] += 2 * speed
        elif direction == Direction.RIGHT:
            self.angles[1] -= 2 * speed
        elif direction == Direction.FORWARD:
            self.angles[2] += 2 * speed
        elif direction == Direction.BACKWARDS:
            self.angles[2] -= 2 * speed

    def transform_vertices(self, camera_pos, camera_angles, project=True, return_np=False):
        t_vertices = []
        for vertex in self.vertices:
            rotation = vertex.rotate(*self.angles)
            rotation.x += camera_pos[0] + self.pos[0]
            rotation.y += camera_pos[1] + self.pos[1]
            rotation.z += camera_pos[2] + self.pos[2]

            rotation = rotation.rotate_yx(*camera_angles)


            if project:
                projection = rotation.project()
            else:
                projection = rotation
            if return_np:
                t_vertices.append(projection.to_arr())
            else:
                t_vertices.append(projection)
        if return_np:
            return np.array(t_vertices)
        else:
            return t_vertices

    def calculate_avg_z(self, vertices):
        avg_z = []
        for idx, face in enumerate(self.faces):
            z = (vertices[face[0]].z +
                 vertices[face[1]].z +
                 vertices[face[2]].z +
                 vertices[face[3]].z) / 4.0
            avg_z.append([idx, z])

        return avg_z


    def get_c_and_norm(self, transformed_vertices, return_d=False):
        normals = []
        centers = []
        for face in self.faces:
            sum_x = 0
            sum_y = 0
            sum_z = 0
            for vertex in face:
                point = transformed_vertices[vertex]
                sum_x += point.x
                sum_y += point.y
                sum_z += point.z
            center = Point3D(sum_x / 4, sum_y / 4, sum_z / 4)
            p1 = transformed_vertices[face[0]].to_arr()
            p2 = transformed_vertices[face[1]].to_arr()
            p3 = transformed_vertices[face[2]].to_arr()
            v1 = p3 - p1
            v2 = p2 - p1
            normal = -np.cross(v1, v2)

            if return_d:
                d = np.dot(normal, p1)
                normals.append(np.array([*normal, d]))
            else:
                normals.append(normal)
            centers.append(center)

        return np.array(centers), np.array(normals)

@jit(nopython=True, fastmath=True)
def bilinear_interpolation(x: np.number, y:np.number, x1: np.number, y1: np.number, q1: np.number, x2: np.number, y2: np.number, q2: np.number, x3: np.number, y3: np.number, q3: np.number):
    w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / \
        ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
    
    if not (np.all(np.isfinite(w1)) and np.all(np.isfinite(w2))):
        print("TOO BAD")
        w1 = np.full_like(w1, 0)
        w2 = np.full_like(w2, 0)
    
    w3 = 1 - w1 - w2
    
    return np.outer(q1, w1) + np.outer(q2, w2) + np.outer(q3, w3)

@jit(nopython=True, fastmath=True)
def area(x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

@jit(nopython=True, fastmath=True, error_model="numpy")
def simple_is_inside_triangle(x: np.number, y: np.number, x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    EPS = 1e-6
    
    A_123 = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    A3 = area(x1, y1, x2, y2, x, y)
    mask = np.abs((A1 + A2 + A3) - A_123) < EPS
    return mask, A1 / A_123, A2 / A_123, A3 / A_123

@jit(nopython=True, fastmath=True, error_model="numpy")
def is_inside_triangle(x: np.number, y: np.number, x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    EPS = 1e-6
    
    A_123 = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    A3 = area(x1, y1, x2, y2, x, y)
    mask = np.abs((A1 + A2 + A3) - A_123) < EPS

    if False:
        mask_inds = np.where(mask)[0]

        L = np.zeros((3, 2))
        A = np.array([x1, y1])
        B = np.array([x2, y2])
        C = np.array([x3, y3])
        L[0] = A
        L[1] = B
        L[2] = C
        
        P = np.zeros((2, x.shape[0]))
        P[0] = x
        P[1] = y

        QR_A, QR_B, QR_C = 0, -1, y


        LP = np.zeros((3, 3))
        AB_A, AB_B, AB_C = (y1 - y2), (x2 - x1), (x1 * y2 - x2 * y1)
        BC_A, BC_B, BC_C = (y2 - y3), (x3 - x2), (x2 * y3 - x3 * y2)
        CA_A, CA_B, CA_C = (y3 - y1), (x1 - x3), (x3 * y1 - x1 * y3)
        LP[0] = [AB_A, AB_B, AB_C]
        LP[1] = [BC_A, BC_B, BC_C]
        LP[2] = [CA_A, CA_B, CA_C]

        I = np.zeros((3, 2, y.shape[0]))
        I[0,0] = (QR_B * AB_C - AB_B * QR_C) / (QR_A * AB_B - AB_A * QR_B)
        I[0,1] = (QR_C * AB_A - AB_C * QR_A) / (QR_A * AB_B - AB_A * QR_B)
        I[1,0] = (QR_B * BC_C - BC_B * QR_C) / (QR_A * BC_B - BC_A * QR_B)
        I[1,1] = (QR_C * BC_A - BC_C * QR_A) / (QR_A * BC_B - BC_A * QR_B)
        I[2,0] = (QR_B * CA_C - CA_B * QR_C) / (QR_A * CA_B - CA_A * QR_B)
        I[2,1] = (QR_C * CA_A - CA_C * QR_A) / (QR_A * CA_B - CA_A * QR_B)


        d = (P[0] - I[:, 0, :])
        l = np.zeros(len(y), dtype=np.int32)
        for i in mask_inds:
            m = -1
            for j in range(3):
                if d[j, i] <= -EPS and (m == -1 or d[j, i] > d[m, i]):
                    m = j
            l[i] = m

        r = np.zeros(len(y), dtype=np.int32)
        for i in mask_inds:
            m = -1
            for j in range(3):
                if d[j, i] >= -EPS and (m == -1 or d[j, i] < d[m, i]):
                    m = j
            r[i] = m

        Q = np.zeros((2, y.shape[0]))
        for i in mask_inds:
            Q[:, i] = I[l[i], :, i]
        
        R = np.zeros((2, y.shape[0]))
        for i in mask_inds:
            R[:, i] = I[r[i], :, i]


        def norm(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2, axis=-1))

        u = np.zeros(len(l))
        w = np.zeros(len(l))
        t = np.zeros(len(l))
        for i in mask_inds:
            u[i] = 1 - norm(L[l[i]], Q.T[i]) / norm(L[l[i]], L[(l[i] + 1) % 3])
            w[i] = 1 - norm(L[r[i]], R.T[i]) / norm(L[r[i]], L[(r[i] + 1) % 3])
            t[i] = 1 - norm(Q.T[i], P.T[i]) / norm(Q.T[i], R.T[i])

        W = np.zeros((3, len(x)))
        for i in mask_inds:
            W[l[i], i] += u[i] * t[i]
            W[(l[i] + 1) % 3, i] += (1 - u[i]) * t[i]
            W[r[i], i] += w[i] * (1 - t[i])
            W[(r[i] + 1) % 3, i] += (1 - w[i]) * (1 - t[i])

        WA = W[0]
        WB = W[1]
        WC = W[2]

        
        return mask, WA, WB, WC
    else:
        return mask, A1 / A_123, A2 / A_123, A3 / A_123

@jit(nopython=True, fastmath=True)
def get_triangle_points(x1: np.number, y1: np.number, x2: np.number, y2:np.number, x3: np.number, y3: np.number):
    xs = np.array([x1, x2, x3], dtype=np.int32)
    ys = np.array([y1, y2, y3], dtype=np.int32)
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    x_delta = x_max - x_min + 1
    y_delta = y_max - y_min + 1

    #Numba недовольна если возвращать None, так что костыль
    if x_max < 0 or x_min > SCREEN_WIDTH or y_max < 0 or y_min > SCREEN_HEIGHT:
        x_delta = y_delta = 1

    if x_min == x_max or y_min == y_max:
        x_delta = y_delta = 1

    X = np.repeat(np.arange(x_delta, dtype=np.int32), y_delta)
    Y = np.array([*np.arange(y_delta, dtype=np.int32)] * x_delta)

    screen_mask = ((x_min + X) > 0) & ((x_min + X) < SCREEN_WIDTH) & (
        (y_min + Y) > 0) & ((y_min + Y) < SCREEN_HEIGHT)
    X = X[screen_mask]
    Y = Y[screen_mask]

    XY_mask, W1, W2, W3 = is_inside_triangle(X, Y, 
                                x1 - x_min,
                                y1 - y_min,
                                x2 - x_min,
                                y2 - y_min,
                                x3 - x_min,
                                y3 - y_min)

    W = np.zeros((3, W1.shape[0]))
    W[0] = W1
    W[1] = W2
    W[2] = W3
    
    X = X[XY_mask]
    Y = Y[XY_mask]
    W = W[:, XY_mask]
    

    X = X + x_min
    Y = Y + y_min

    return X, Y, W

@jit(nopython=True, fastmath=True)
def screen_space(draw_buffer, x, y):
    mask = (x >= 0) & (y >= 0) & (x < draw_buffer.shape[0]) & (y < draw_buffer.shape[1])
    return mask


@jit(nopython=True, fastmath=True)
def draw_wireframe(draw_buffer, z_buffer, triangle, color, force_z=False):
    for i in range(len(triangle)):
        x1, y1, z1, _v1 = triangle[i]
        x2, y2, z2, _v2 = triangle[(i + 1) % len(triangle)]
        l = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        x = np.linspace(x1, x2, l)
        y = np.linspace(y1, y2, l)
        z = np.linspace(z1, z2, l)

        screen_mask = screen_space(draw_buffer, x, y)
        x = x[screen_mask].astype(np.int32)
        y = y[screen_mask].astype(np.int32)
        z = z[screen_mask].astype(np.int32)

        if force_z:
            z = np.full_like(z, 0)

        for _x, _y, _z in zip(x, y, z):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = color
                z_buffer[_x, _y] = _z
            #draw_buffer[_x, _y] = color

@jit(nopython=True, fastmath=True)
def draw_lumbert(draw_buffer, z_buffer, triangle, color):
    XYW = get_triangle_points(triangle[0, 0], triangle[0,1], triangle[1,0], triangle[1,1], triangle[2,0], triangle[2,1])
    if XYW is not None and len(XYW[0]) > 0:
        X, Y, W = XYW
        Z = triangle[0, 2] * W[0] + triangle[1, 2] * W[1] + triangle[2, 2] * W[2]

        for _x, _y, _z in zip(X, Y, Z):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = color
                z_buffer[_x, _y] = _z

@jit(nopython=True, fastmath=True)
def draw_gouraud(draw_buffer, z_buffer, triangle, colors):
    XYW = get_triangle_points(triangle[0, 0], triangle[0,1], triangle[1,0], triangle[1,1], triangle[2,0], triangle[2,1])
    if XYW is not None and len(XYW[0]) > 0:
        X, Y, W = XYW
        
        Z = triangle[0, 2] * W[0] + triangle[1, 2] * W[1] + triangle[2, 2] * W[2]
        C = np.outer(W[0], colors[int(triangle[0, 3])]) + np.outer(W[1], colors[int(triangle[1, 3])]) + np.outer(W[2], colors[int(triangle[2, 3])])

        for _x, _y, _z, _c in zip(X, Y, Z, C):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = _c
                z_buffer[_x, _y] = _z

@jit(nopython=True, fastmath=True)
def draw_phong(draw_buffer, z_buffer, triangle, plane, other_planes, lights):
    #1 / 1.2246832243
    phong_coeff = FOCAL_LENGTH / SCREEN_WIDTH 

    x_min = np.min(triangle[:,0]) / SCREEN_WIDTH - 0.5
    y_min = np.min(triangle[:,1]) / SCREEN_HEIGHT - 0.5
    x_max = np.max(triangle[:,0]) / SCREEN_WIDTH - 0.5
    y_max = np.max(triangle[:,1]) / SCREEN_HEIGHT - 0.5

    ray_origin = np.array([0, 0, 0])

    SCREEN_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
    H, W = 1 / SCREEN_RATIO, 1

    xxx = np.linspace(-W / 2, W / 2, SCREEN_WIDTH)
    yyy = np.linspace(-H / 2, H / 2, SCREEN_HEIGHT)
    
    OFFSET = 0.1
    x_min -= OFFSET
    x_max += OFFSET
    y_min -= OFFSET
    y_max += OFFSET

    x_min, x_max = -x_max, -x_min
    
    y_min *= 1 / SCREEN_RATIO
    y_max *= 1 / SCREEN_RATIO
    y_min, y_max = -y_max, -y_min

    xxx = xxx[(xxx > x_min) & (xxx < x_max)]
    yyy = yyy[(yyy > y_min) & (yyy < y_max)]

    X = np.repeat(xxx, len(yyy))
    Y = np.array([*yyy] * len(xxx))

    x1, y1, z1 = ray_origin
    x2 = X
    y2 = Y
    z2 = -phong_coeff

    a, b, c, d = plane

    ts = (-d - a * x1 - b * y1 - c * z1) / (a * (x2 - x1) + b * (y2 - y1) + c * (z2 - z1))

    xs = ts * (x2 - x1) + x1
    ys = ts * (y2 - y1) + y1
    zs = ts * (z2 - z1) + z1

    flags = ts > 0
    for plane1 in other_planes:
        a1, b1, c1, d1 = plane1

        flags &= (
            (a1 * xs + b1 * ys + c1 * zs + d1) > 0.01)

    xs = xs[flags]
    ys = ys[flags]
    zs = zs[flags]

    if (len(xs) == 0):
        return

    diffuse = np.zeros((len(xs), 3))
    specular = np.zeros((len(xs), 3))
    for light in lights:
        light_pos = -light[:3]
        xyz = np.zeros((3,len(xs)))
        xyz[0] = xs
        xyz[1] = ys
        xyz[2] = zs
        l = light_pos - xyz.T

        r = 2 * -plane[:3] - l

        l_norm = np.sqrt(np.sum(l ** 2, axis=1))
        l[:, 0] /= l_norm
        l[:, 1] /= l_norm
        l[:, 2] /= l_norm

        d = np.dot(-plane[:3], l.T)
        o = np.outer(d, light[3:])

        diffuse = diffuse + np.clip(o * 0.5, 0, 1) * 255

        r_norm = np.sqrt(np.sum(r ** 2, axis=1))
        r[:, 0] /= r_norm
        r[:, 1] /= r_norm
        r[:, 2] /= r_norm

        specular_coef = -xyz.T
        specular_coef_norm = np.sqrt(np.sum(r ** 2, axis=1))
        specular_coef[:, 0] /= specular_coef_norm
        specular_coef[:, 1] /= specular_coef_norm
        specular_coef[:, 2] /= specular_coef_norm

        #print(r, specular_coef)

        wee_wee_1 = r * specular_coef

        #print(wee_wee_1)
        #specular = specular + np.clip(np.power(np.maximum(wee_wee_1, 0), 1) * 0.1, 0, 1) * 255 

    xs = ((X + (W * 0.5)) * SCREEN_WIDTH / W).astype(np.int32)
    ys = ((Y + (H * 0.5)) * SCREEN_HEIGHT / H).astype(np.int32)

    xs = SCREEN_WIDTH - xs[flags] - 1
    ys = SCREEN_HEIGHT - ys[flags] - 1
    
    for _x, _y, _z, _k in zip(xs, ys, zs, diffuse + specular):
        _z = -_z
        if z_buffer[_x, _y] > _z:
            draw_buffer[_x, _y] = _k
            z_buffer[_x, _y] = _z


@jit(nopython=True, fastmath=True)
def draw_phong2(draw_buffer, z_buffer, triangle, plane, color, lights):
    XYW = get_triangle_points(triangle[0, 0], triangle[0,1], triangle[1,0], triangle[1,1], triangle[2,0], triangle[2,1])
    if XYW is not None and len(XYW[0]) > 0:
        X, Y, W = XYW
        Z = triangle[0, 2] * W[0] + triangle[1, 2] * W[1] + triangle[2, 2] * W[2]
        
        xs = X
        ys = Y

        factor = FOCAL_LENGTH / Z
        X = (X - SCREEN_WIDTH / 2) / factor
        Y = (Y - SCREEN_HEIGHT / 2) / factor
        
        AMBIENT_COEFF = 0.05
        DIFFUSE_COEFF = 1
        SPECULAR_COEFF = 0.2

        diffuse = np.zeros((len(X), 3))
        specular = np.zeros((len(X), 3))
        for light in lights:
            light_color = light[3:]
            light_pos = light[:3]
            xyz = np.zeros((3,len(xs)))
            xyz[0] = X
            xyz[1] = Y
            xyz[2] = Z
            l = light_pos - xyz.T
            
            n = plane[:3]
            n_norm = np.sqrt(np.sum(n ** 2))
            n /= n_norm

            l_norm = np.sqrt(np.sum(l ** 2, axis=1))
            l[:, 0] /= l_norm
            l[:, 1] /= l_norm
            l[:, 2] /= l_norm

            r = 2 * n - l
            r_norm = np.sqrt(np.sum(r ** 2, axis=1))
            r[:, 0] /= r_norm
            r[:, 2] /= r_norm
            r[:, 1] /= r_norm

            d = np.outer(np.maximum(np.dot(n, l.T), 0), light_color)
            diffuse += d * DIFFUSE_COEFF

            if np.max(d) < 0.001:
                continue

            neg_i = -xyz.T
            neg_i_norm = np.sqrt(np.sum(neg_i ** 2, axis=1))
            neg_i[:, 0] /= neg_i_norm
            neg_i[:, 1] /= neg_i_norm
            neg_i[:, 2] /= neg_i_norm

            specular += np.outer(np.power(np.maximum(np.sum(r * neg_i, axis=1), 0), 50), light_color) * SPECULAR_COEFF

            
        c = np.clip(color * (AMBIENT_COEFF + diffuse + specular), 0, 1) * 255

        for _x, _y, _z, _c in zip(xs, ys, Z, c):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = _c
                z_buffer[_x, _y] = _z



class Simulation():
    def __init__(self, screen_width, screen_height):
        pygame.init()
        pygame.display.set_caption('BumbleBee Flight')
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.current_shading = Shading.WIREFRAME
        self._objects = []

        self.camera_pos = np.zeros(3)
        self.camera_pos[2] = 5
        self.camera_angles = np.zeros(3)
        self.selected_object = None
        self.prev_keys = None


    def create_cube(self):
        # if len(self._objects) > 2:
        #     return
        self._objects.append(Cube())

    def run(self):
        a = 1.5
        d = 1
        point_lights = [
            # PointLight(Point3D(1000, 1000, 0), [0, 1, 0]),
            # PointLight(Point3D(-1000, 1000, 0), [0, 0, 1]),
            # PointLight(Point3D(0, -1000, 0), [1, 0, 0]),
            PointLight(Point3D(0, a, d), np.array([1, 0, 0])),
            PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), np.array([0, 1, 0])),
            PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), np.array([0, 0, 1])),

            #PointLight(Point3D(0, a, d), [1, 0, 0]),
            # PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            # PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), [0, 0, 1]),
            # PointLight(Point3D(-a / 2, np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            # PointLight(Point3D(a / 2, np.sqrt(3) / 2 * a, d), [0, 0, 1]),

            #PointLight(Point3D(1, 1, 10), [1, 1, 1])
            #PointLight(Point3D(0, 0, -1000), [1, 1, 1])
            ]
        

        SIZE = 1
        for i in range(-SIZE, SIZE + 1):
            for j in range(-SIZE, SIZE + 1):
                cube = Cube()
                cube.translate_cube(i * 2, j * 2, 5)
                #cube.rotate_cube(Direction.DOWN, 15)
                self._objects.append(cube)
        
        # cube = Cube()
        # cube.translate_cube(0, 0, 5)
        # self._objects.append(cube)



        while True:
            self.clock.tick(FPS)
            events = pygame.event.get()
            keys = pygame.key.get_pressed()
            if self.prev_keys is None:
                self.prev_keys = keys


            if pygame.key.get_mods() > 0:
                mouse_move = (0, 0)
            else:
                mouse_move = pygame.mouse.get_rel()
                pygame.mouse.set_pos([SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2]) 

            if np.max(np.abs(np.array(mouse_move))) < 100:
                self.camera_angles += np.array([mouse_move[1], -mouse_move[0], 0]) * 1 * (1 / FPS)

            mouse_click = False

            for event in events:
                if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                    pygame.quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_click = True

            draw_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 0)
            z_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), np.inf)

            new_selected_obj = None

            for light in point_lights:
                cube = Cube(0.1)
                cube.translate_cube(*light.point.to_arr())
                triangles = cube.draw_cube(self.camera_pos, self.camera_angles, [])
                _, normals = cube.get_c_and_norm(cube.transform_vertices(self.camera_pos, self.camera_angles, False), True)
                for polygon in triangles:
                    triangle = polygon[0]
                    draw_lumbert(draw_buffer, z_buffer, triangle, np.array(light.color) * 255)
                    if mouse_click and simple_is_inside_triangle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, *triangle[:, :2].reshape(-1))[0]:
                        a, b, c, d = normals[face_ind]
                        t = d / c
                        if t > 0 and (new_selected_obj is None or new_selected_obj[0] > t):
                            new_selected_obj = (t, light)
                    if self.selected_object == light:
                        draw_wireframe(draw_buffer, z_buffer, triangle, (1 - light.color) * 255, True)

            MOVE_SPEED = 1 * (1 / FPS)
            ROTATION_SPEED = 10 * (1 / FPS)

            for obj_ind, obj in enumerate(self._objects):
                triangles = obj.draw_cube(self.camera_pos, self.camera_angles, point_lights, self.current_shading, triangles=self.current_shading != Shading.PHONG_RAYS)
                _, normals = obj.get_c_and_norm(obj.transform_vertices(self.camera_pos, self.camera_angles, False), True)

                for polygon in triangles:
                    triangle = polygon[0]
                    face_ind = polygon[1]
                    #draw_wireframe(draw_buffer, triangle)
                    if self.current_shading == Shading.WIREFRAME:
                        draw_wireframe(draw_buffer, z_buffer, triangle, obj.color * 255)
                    elif self.current_shading == Shading.LAMBERT:
                        draw_lumbert(draw_buffer, z_buffer, triangle, obj.lambert_colors[face_ind])
                    elif self.current_shading == Shading.GOURAUD:
                        draw_gouraud(draw_buffer, z_buffer, triangle, obj.gouraud_colors)
                    elif self.current_shading == Shading.PHONG or self.current_shading == Shading.PHONG_RAYS:
                        lights = np.zeros((len(point_lights), 6))
                        for i, light in enumerate(point_lights):
                            light_pos = Point3D(*light.point.to_arr())
                            light_pos.x += self.camera_pos[0]
                            light_pos.y += self.camera_pos[1]
                            light_pos.z += self.camera_pos[2]
                            rotation = light_pos.rotate_yx(*self.camera_angles)
                            lights[i] = [*rotation.to_arr(), *light.color]
                        if self.current_shading == Shading.PHONG:
                            draw_phong2(draw_buffer, z_buffer, triangle, normals[face_ind], obj.color, lights)
                        else:
                            draw_phong(draw_buffer, z_buffer, triangle, normals[face_ind], normals[np.arange(len(normals)) != face_ind], lights)

                    if mouse_click and simple_is_inside_triangle(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, *triangle[:, :2].reshape(-1))[0]:
                        a, b, c, d = normals[face_ind]
                        t = d / c
                        if t > 0 and (new_selected_obj is None or new_selected_obj[0] > t):
                            new_selected_obj = (t, obj)

                    if obj == self.selected_object:
                        draw_wireframe(draw_buffer, z_buffer, triangle, obj.color * 255, True)
                    

                if new_selected_obj:
                    self.selected_object = new_selected_obj[1]


                objs = [self.selected_object] if self.selected_object is not None else self._objects
                for obj in objs:
                    if keys[pygame.K_t]:
                        obj.rotate_cube(Direction.UP, ROTATION_SPEED)
                    elif keys[pygame.K_g]:
                        obj.rotate_cube(Direction.DOWN, ROTATION_SPEED)
                    elif keys[pygame.K_h]:
                        obj.rotate_cube(Direction.LEFT, ROTATION_SPEED)
                    elif keys[pygame.K_f]:
                        obj.rotate_cube(Direction.RIGHT, ROTATION_SPEED)
                    elif keys[pygame.K_r]:
                        obj.rotate_cube(Direction.FORWARD, ROTATION_SPEED)
                    elif keys[pygame.K_y]:
                        obj.rotate_cube(Direction.BACKWARDS, ROTATION_SPEED)
                    if keys[pygame.K_k]:
                        obj.translate_cube(0, 0, MOVE_SPEED)
                    elif keys[pygame.K_i]:
                        obj.translate_cube(0, 0, -MOVE_SPEED)
                    elif keys[pygame.K_l]:
                        obj.translate_cube(MOVE_SPEED, 0, 0)
                    elif keys[pygame.K_j]:
                        obj.translate_cube(-MOVE_SPEED, 0, 0)
                    elif keys[pygame.K_u]:
                        obj.translate_cube(0, MOVE_SPEED, 0)
                    elif keys[pygame.K_o]:
                        obj.translate_cube(0, -MOVE_SPEED, 0)

                    if keys[pygame.K_z]:
                        obj.color = np.array([1, 1, 1])
                    elif keys[pygame.K_x]:
                        obj.color = np.array([0, 0, 1])
                    elif keys[pygame.K_c]:
                        obj.color = np.array([0, 1, 0])
                    elif keys[pygame.K_v]:
                        obj.color = np.array([1, 0, 0])
                    elif keys[pygame.K_b]:
                        obj.color = np.array([0, 1, 1])
                    elif keys[pygame.K_n]:
                        obj.color = np.array([1, 0, 1])
                    elif keys[pygame.K_m]:
                        obj.color = np.array([1, 1, 0])
                    elif keys[pygame.K_COMMA] and not self.prev_keys[pygame.K_COMMA]:
                        obj.color = np.random.random(3)
                    elif keys[pygame.K_PERIOD]:
                        obj.color = np.array([0, 0, 0])

            if keys[pygame.K_LCTRL]:
                self.selected_object = None

            if keys[pygame.K_SPACE] and not self.prev_keys[pygame.K_SPACE]:
                self.create_cube()

            elif keys[pygame.K_1]:
                self.current_shading = Shading.WIREFRAME
            elif keys[pygame.K_2]:
                self.current_shading = Shading.LAMBERT
            elif keys[pygame.K_3]:
                self.current_shading = Shading.GOURAUD
            elif keys[pygame.K_4]:
                self.current_shading = Shading.PHONG
            elif keys[pygame.K_5]:
                self.current_shading = Shading.PHONG_RAYS

            forward = Point3D(0, 0, -MOVE_SPEED).rotate_xy(*-self.camera_angles).to_arr()
            right = Point3D(-MOVE_SPEED, 0, 0).rotate_xy(*-self.camera_angles).to_arr()
            top = Point3D(0, -MOVE_SPEED, 0).rotate_xy(*-self.camera_angles).to_arr()

            if keys[pygame.K_w]:
                self.camera_pos += forward * 5
            if keys[pygame.K_s]:
                self.camera_pos -= forward * 5
            if keys[pygame.K_d]:
                self.camera_pos += right * 5
            if keys[pygame.K_a]:
                self.camera_pos -= right * 5
            if keys[pygame.K_q]:
                self.camera_pos += top * 5
            if keys[pygame.K_e]:
                self.camera_pos -= top * 5

            surf = pygame.surfarray.make_surface(draw_buffer)
            #surf = pygame.surfarray.make_surface(np.repeat(z_buffer[:, :, np.newaxis], 3, -1) * 20)
            self.screen.blit(surf, (0, 0))

            font = pygame.font.Font(None, 26)
            fps_text = font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, Color.WHITE.value)
            place = fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)

            self.prev_keys = keys
            pygame_widgets.update(events)
            pygame.display.update()


Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
