import pygame
from pygame import draw
import pygame_widgets
from pygame_widgets.button import Button, ButtonArray
import numpy as np
from enum import Enum
from numba import jit
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


SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 512

FOV = np.radians(90)

FOCAL_LEGNTH = SCREEN_WIDTH / 2 * np.cos(FOV / 2) / np.sin(FOV / 2)

FPS = 60


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

    def rotate(self, a, b, g):
        a, b, g = a * np.pi / 180, b * np.pi / 180, g * np.pi / 180
        sin_a, cos_a = np.sin(a), np.cos(a)
        sin_b, cos_b = np.sin(b), np.cos(b)
        sin_g, cos_g = np.sin(g), np.cos(g)

        matrix = np.array([
            [cos_b * cos_g , sin_a * sin_b * cos_g - cos_a * sin_g, cos_a * sin_b * cos_g + sin_a * sin_g],
            [cos_b * sin_g, sin_a * sin_b * sin_g + cos_a * cos_g, cos_a * sin_b * sin_g - sin_a * cos_g],
            [-sin_b, sin_a * cos_b, cos_a * cos_b]
        ])
        x, y, z = np.dot(matrix, np.array([self.x, self.y, self.z]))
        return Point3D(x, y, z)

    def project(self):
        if (self.z) > 0:
            factor = FOCAL_LEGNTH / (self.z)
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
    return intensities

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
            intensities[i] += 1 * light_color * intensity_coef

    return intensities

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

    def rotate_cube(self, direction):
        if direction == Direction.UP:
            self.angles[0] += 2
        elif direction == Direction.DOWN:
            self.angles[0] -= 2
        elif direction == Direction.LEFT:
            self.angles[1] += 2
        elif direction == Direction.RIGHT:
            self.angles[1] -= 2
        elif direction == Direction.FORWARD:
            self.angles[2] += 2
        elif direction == Direction.BACKWARDS:
            self.angles[2] -= 2

    def transform_vertices(self, canera_pos, camera_angles, project=True, return_np=False):
        t_vertices = []
        for vertex in self.vertices:
            rotation = vertex.rotate(*self.angles)
            rotation.x += canera_pos[0] + self.pos[0]
            rotation.y += canera_pos[1] + self.pos[1]
            rotation.z += canera_pos[2] + self.pos[2]
            rotation = rotation.rotate(*camera_angles)
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
    return np.abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                    + x3 * (y1 - y2)) / 2.0)

@jit(nopython=True, fastmath=True)
def is_inside_triangle(x: np.number, y: np.number, x1: np.number, y1: np.number, x2: np.number, y2: np.number, x3: np.number, y3: np.number):
    A = area(x1, y1, x2, y2, x3, y3)
    A1 = area(x, y, x2, y2, x3, y3)
    A2 = area(x1, y1, x, y, x3, y3)
    A3 = area(x1, y1, x2, y2, x, y)
    return (A1 + A2 + A3) - A < 0.001, A1 / A, A2 / A, A3 / A

@jit(nopython=True, fastmath=True)
def get_triangle_points(x1: np.number, y1: np.number, x2: np.number, y2:np.number, x3: np.number, y3: np.number):
    xs = np.array([x1, x2, x3])
    ys = np.array([y1, y2, y3])
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

    X = np.repeat(np.arange(x_delta), y_delta)
    Y = np.array([*np.arange(y_delta)] * x_delta)

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
    return x[mask].astype(np.int32), y[mask].astype(np.int32)

@jit(nopython=True, fastmath=True)
def draw_wireframe(draw_buffer, triangle):
    for i in range(len(triangle)):
        x1, y1, _z1, _v1 = triangle[i]
        x2, y2, _z1, _v2 = triangle[(i + 1) % len(triangle)]
        l = int(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
        x = np.linspace(x1, x2, l)
        y = np.linspace(y1, y2, l)
        x, y = screen_space(draw_buffer, x, y)

        for _x, _y in zip(x, y):
            draw_buffer[_x, _y] = [255, 255, 255]

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
    phong_coeff = FOCAL_LEGNTH / SCREEN_WIDTH 

    x_min = np.min(triangle[:,0]) / SCREEN_WIDTH - 0.5
    y_min = np.min(triangle[:,1]) / SCREEN_HEIGHT - 0.5
    x_max = np.max(triangle[:,0]) / SCREEN_WIDTH - 0.5
    y_max = np.max(triangle[:,1]) / SCREEN_HEIGHT - 0.5

    ray_origin = np.array([0, 0, 0])
    # range(0, SCREEN_WIDTH, 10):

    SCREEN_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
    H, W = 1 / SCREEN_RATIO, 1   # SCREEN_HEIGHT / FOV / 2, SCREEN_WIDTH / FOV / 2

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

        factor = FOCAL_LEGNTH / Z
        X = (X - SCREEN_WIDTH / 2) / factor
        Y = (Y - SCREEN_HEIGHT / 2) / factor
        
        AMBIENT_COEFF = 0.05
        DIFFUSE_COEFF = 1
        SPECULAR_COEFF = 0.5

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

            #print(np.max(np.sum(r * neg_i, axis=1)))

            specular += np.outer(np.power(np.maximum(np.sum(r * neg_i, axis=1), 0), 50), light_color) * SPECULAR_COEFF



        for _x, _y, _z, _d, _s in zip(xs, ys, Z, diffuse, specular):
            if z_buffer[_x, _y] > _z:
                draw_buffer[_x, _y] = np.clip(color * (AMBIENT_COEFF + _d + _s), 0, 1) * 255
                z_buffer[_x, _y] = _z



class Simulation():
    def __init__(self, screen_width, screen_height, objects=[]):
        pygame.init()
        pygame.display.set_caption('BumbleBee Flight')
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.current_mode = Mode.ROTATING
        self.current_shading = Shading.WIREFRAME
        self._objects = objects

        self.camera_pos = np.zeros(3)
        self.camera_pos[2] = 5
        self.camera_angles = np.zeros(3)


    def add_cube_btn(self):
        button = Button(
            # Mandatory Parameters
            self.screen,  # Surface to place button on
            SCREEN_WIDTH - 100,  # X-coordinate of top left corner
            0,  # Y-coordinate of top left corner
            100,  # Width
            50,  # Height

            # Optional Parameters
            text='Добавить куб',  # Text to display
            fontSize=20,  # Size of font
            inactiveColour=(0, 250, 154),
            hoverColour=(128, 128, 128),
            pressedColour=(105, 105, 105),
            onClick=self.create_cube  # Function to call when clicked on
        )

    def create_cube(self):
        # if len(self._objects) > 2:
        #     return
        self._objects.append(Cube())

    def run(self):
        self.add_cube_btn()
        a = 1
        d = 3
        point_lights = [
            # PointLight(Point3D(1000, 1000, 0), [0, 1, 0]),
            # PointLight(Point3D(-1000, 1000, 0), [0, 0, 1]),
            # PointLight(Point3D(0, -1000, 0), [1, 0, 0]),
            PointLight(Point3D(0, a, d), [1, 0, 0]),
            PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), [0, 0, 1]),

            #PointLight(Point3D(0, a, d), [1, 0, 0]),
            # PointLight(Point3D(-a / 2, -np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            # PointLight(Point3D(a / 2, -np.sqrt(3) / 2 * a, d), [0, 0, 1]),
            # PointLight(Point3D(-a / 2, np.sqrt(3) / 2 * a, d), [0, 1, 0]),
            # PointLight(Point3D(a / 2, np.sqrt(3) / 2 * a, d), [0, 0, 1]),

            #PointLight(Point3D(1, 1, 10), [1, 1, 1])
            ]
        

        SIZE = 1
        for i in range(-SIZE, SIZE + 1):
            for j in range(-SIZE, SIZE + 1):
                cube = Cube()
                cube.translate_cube(i * 2, j * 2, 5)
                self._objects.append(cube)
        

        while True:
            self.clock.tick(FPS)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
            keys = pygame.key.get_pressed()
            draw_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 0)
            z_buffer = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), np.inf)

            for light in point_lights:
                cube = Cube(0.1)
                cube.translate_cube(*light.point.to_arr())
                triangles = cube.draw_cube(self.camera_pos, self.camera_angles, [])
                for polygon in triangles:
                    triangle = polygon[0]
                    draw_lumbert(draw_buffer, z_buffer, triangle, np.array(light.color) * 255)

            
            for obj_ind, obj in enumerate(self._objects):
                triangles = obj.draw_cube(self.camera_pos, self.camera_angles, point_lights, self.current_shading, triangles=self.current_shading != Shading.PHONG_RAYS)

                _, normals = obj.get_c_and_norm(
                    obj.transform_vertices(self.camera_pos, self.camera_angles, False), True)

                for polygon in triangles:
                    triangle = polygon[0]
                    face_ind = polygon[1]
                    #draw_wireframe(draw_buffer, triangle)
                    if self.current_shading == Shading.WIREFRAME:
                        draw_wireframe(draw_buffer, triangle)
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
                            rotation = light_pos.rotate_X(self.camera_angles[0]).rotate_Y(self.camera_angles[1]).rotate_Z(self.camera_angles[2])
                            lights[i] = [*rotation.to_arr(), *light.color]
                        if self.current_shading == Shading.PHONG:
                            draw_phong2(draw_buffer, z_buffer, triangle, normals[face_ind], obj.color, lights)
                        else:
                            draw_phong(draw_buffer, z_buffer, triangle, normals[face_ind], normals[np.arange(len(normals)) != face_ind], lights)
                        

                if keys[pygame.K_z]:
                    self.current_mode = Mode.ROTATING
                elif keys[pygame.K_x]:
                    self.current_mode = Mode.MOVING
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
                elif keys[pygame.K_6]:
                    self.phong_coeff += 0.01
                    print(self.phong_coeff)
                elif keys[pygame.K_7]:
                    self.phong_coeff -= 0.01
                    print(self.phong_coeff)

                if self.current_mode == Mode.ROTATING:
                    if keys[pygame.K_w]:
                        obj.rotate_cube(Direction.UP)
                    elif keys[pygame.K_s]:
                        obj.rotate_cube(Direction.DOWN)
                    elif keys[pygame.K_a]:
                        obj.rotate_cube(Direction.LEFT)
                    elif keys[pygame.K_d]:
                        obj.rotate_cube(Direction.RIGHT)
                    elif keys[pygame.K_q]:
                        obj.rotate_cube(Direction.FORWARD)
                    elif keys[pygame.K_e]:
                        obj.rotate_cube(Direction.BACKWARDS)
                elif self.current_mode == Mode.MOVING:
                    if keys[pygame.K_w]:
                        obj.translate_cube(0, 0, 0.05)
                    elif keys[pygame.K_s]:
                        obj.translate_cube(0, 0, -0.05)
                    elif keys[pygame.K_a]:
                        obj.translate_cube(-0.05, 0, 0)
                    elif keys[pygame.K_d]:
                        obj.translate_cube(0.05, 0, 0)
                    elif keys[pygame.K_q]:
                        obj.translate_cube(0, -0.05, 0)
                    elif keys[pygame.K_e]:
                        obj.translate_cube(0, 0.05, 0)




            forward = Point3D(0, 0, -1).rotate_X(-self.camera_angles[0]).rotate_Y(self.camera_angles[1]).rotate_Z(-self.camera_angles[2])
            right = Point3D(-1, 0, 0).rotate_X(-self.camera_angles[0]).rotate_Y(self.camera_angles[1]).rotate_Z(-self.camera_angles[2])
            top = Point3D(0, -1, 0).rotate_X(-self.camera_angles[0]).rotate_Y(self.camera_angles[1]).rotate_Z(-self.camera_angles[2])

            #self.camera_angles[2] = 0

            forward = Point3D(0, 0, 1).rotate(-self.camera_angles[0], -self.camera_angles[1], -self.camera_angles[2])
            right = Point3D(1, 0, 0).rotate(-self.camera_angles[0], -self.camera_angles[1], -self.camera_angles[2])
            top = Point3D(0, 1, 0).rotate(-self.camera_angles[0], -self.camera_angles[1], -self.camera_angles[2])

            forward = forward.to_arr()
            right = right.to_arr()
            top = top.to_arr()


            print(self.camera_pos)

            print(forward, right, top, self.camera_angles)

            if keys[pygame.K_t]:
                self.camera_angles[0] -= 3
            if keys[pygame.K_g]:
                self.camera_angles[0] += 3
            if keys[pygame.K_f]:
                self.camera_angles[1] -= 3
            if keys[pygame.K_h]:
                self.camera_angles[1] += 3
            if keys[pygame.K_r]:
                self.camera_angles[2] -= 3
            if keys[pygame.K_y]:
                self.camera_angles[2] += 3

            #self.camera_angles[2] = 0

            if keys[pygame.K_i]:
                self.camera_pos += 0.5 * forward
            if keys[pygame.K_k]:
                self.camera_pos -= 0.5 * forward
            if keys[pygame.K_l]:
                self.camera_pos += 0.5 * right
            if keys[pygame.K_j]:
                self.camera_pos -= 0.5 * right
            if keys[pygame.K_u]:
                self.camera_pos += 0.5 * top
            if keys[pygame.K_o]:
                self.camera_pos -= 0.5 * top


            # z_buffer_mask = np.isfinite(z_buffer)
            # if (np.sum(z_buffer_mask) > 0):
            #     print(np.min(z_buffer[z_buffer_mask]), np.max(z_buffer[z_buffer_mask]))
            #     #z_buffer[z_buffer_mask] += np.min(z_buffer[z_buffer_mask])
            #     #z_buffer[z_buffer_mask] /= np.max(z_buffer[z_buffer_mask])
            #     z_buffer[z_buffer_mask] *= 50

            # draw_buffer[:, :, 0] = z_buffer
            # draw_buffer[:, :, 1] = z_buffer
            # draw_buffer[:, :, 2] = z_buffer
            surf = pygame.surfarray.make_surface(draw_buffer)
            self.screen.blit(surf, (0, 0))

            font = pygame.font.Font(None, 26)
            fps_text = font.render(
                f'FPS: {np.round(self.clock.get_fps())}', True, Color.WHITE.value)
            place = fps_text.get_rect(
                center=(SCREEN_WIDTH - 50, SCREEN_HEIGHT - 20))
            self.screen.blit(fps_text, place)

            pygame_widgets.update(events)
            pygame.display.update()


Simulation(SCREEN_WIDTH, SCREEN_HEIGHT).run()
