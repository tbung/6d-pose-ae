# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Modified by Till Bungert
# -----------------------------------------------------------------------------
import numpy as np
from glumpy import app, gl, glm, gloo
from glumpy.ext import png


def create_cube():
    vtype = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3), ('a_color',    np.float32, 4)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                  [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1]],
                 dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0],
                  [-1, 0, 1], [0, -1, 0], [0, 0, -1]])
    # Vertice colors
    c = np.array([[0.3, 0.3, 0.3, 1], [0.3, 0.3, 0.3, 1], [0.3, 0.3, 0.3, 1],
                  [0.3, 0.3, 0.3, 1],
                  [0.3, 0.3, 0.3, 1], [0.3, 0.3, 0.3, 1], [0.3, 0.3, 0.3, 1],
                  [0.3, 0.3, 0.3, 1]])
    # Texture coords
    t = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_c = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]
    faces_t = [0, 1, 2, 3,  0, 1, 2, 3,   0, 1, 2, 3,
               3, 2, 1, 0,  0, 1, 2, 3,   0, 1, 2, 3]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal'] = n[faces_n]
    vertices['a_color'] = c[faces_c]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6 * (2 * 3))
    filled += np.repeat(4 * np.arange(6, dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)

    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)
    outline = outline.view(gloo.IndexBuffer)

    return vertices, filled, outline


def checkerboard(grid_num=8, grid_size=32):
    """ Checkerboard pattern """

    row_even = grid_num // 2 * [0, 1]
    row_odd = grid_num // 2 * [1, 0]
    Z = np.row_stack(grid_num // 2 * (row_even, row_odd)).astype(np.uint8)
    return 255 * Z.repeat(grid_size, axis=0).repeat(grid_size, axis=1)


def mono():
    return 255 * np.ones((32, 32))


class CubeRenderer:
    def __init__(self):
        with open('cube.vert') as f:
            self.vertex = f.read()

        with open('cube.frag') as f:
            self.fragment = f.read()

        self.window = app.Window(width=512, height=512, visible=False,
                                 color=(0.0, 0.0, 0.0, 1.00))
        self.framebuffer = np.zeros((self.window.height, self.window.width *
                                     3), dtype=np.uint8)

        V, I, _ = create_cube()
        self.cube = gloo.Program(self.vertex, self.fragment)
        self.cube.bind(V)

        self.cube["u_light_position"] = 3, 3, 3
        self.cube["u_light_intensity"] = 1, 1, 1
        self.cube["u_light_ambient"] = 0.2
        # cube['u_texture'] = checkerboard()
        self.cube['u_texture'] = mono()
        self.cube['u_model'] = np.eye(4, dtype=np.float32)
        self.cube['u_view'] = glm.translation(0, 0, -10)
        self.min_xy, self.max_xy = -2, 2
        self.frame = 0

        @self.window.event
        def on_draw(dt):
            self.window.clear()

            # Fill cube
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            self.cube['u_color'] = 1, 1, 1, 1
            self.cube.draw(gl.GL_TRIANGLES, I)

            # Rotate cube
            view = self.cube['u_view'].reshape(4, 4)
            model = np.eye(4, dtype=np.float32)
            theta = np.random.random_sample() * 90
            phi = np.random.random_sample() * 180
            glm.rotate(model, theta, 0, 0, 1)
            glm.rotate(model, phi, 0, 1, 0)

            # Translate cube
            x, y = np.random.random_sample(2) * (self.max_xy - self.min_xy) + self.min_xy
            glm.translate(model, x, y, 0)

            self.cube['u_model'] = model
            self.cube['u_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

            # Export screenshot
            if self.frame:  # Skip empty zero frame
                gl.glReadPixels(0, 0, self.window.width, self.window.height,
                                gl.GL_RGB, gl.GL_UNSIGNED_BYTE,
                                self.framebuffer)
                png.from_array(self.framebuffer,
                               'RGB').save(f'cube/{self.frame-1:05d}.png')
            self.frame += 1

        @self.window.event
        def on_resize(width, height):
            self.cube['u_projection'] = glm.perspective(45.0, width /
                                                        float(height), 2.0,
                                                        100.0)

        @self.window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glPolygonOffset(1, 1)
            gl.glEnable(gl.GL_LINE_SMOOTH)

        app.run(framecount=10)


if __name__ == '__main__':
    CubeRenderer()
