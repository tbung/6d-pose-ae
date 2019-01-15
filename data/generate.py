# Adapted from
# https://github.com/glumpy/glumpy/blob/master/examples/tutorial/light-cube-simple.py
import numpy as np
from glumpy import app, gl, glm, gloo
from glumpy.ext import png
from pathlib import Path
import fire
import tqdm
import logging
import trimesh
np.set_printoptions(threshold=np.inf)

log = logging.getLogger('glumpy')
log.setLevel(logging.WARNING)

shapes = ['cat', 'eggbox']


def load_ply(name):
    vtype = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3), ('a_color',    np.float32, 4)]

    mesh = trimesh.load(f"./data/{name}/mesh.ply")
    vertices = np.zeros(mesh.vertices.shape[0], dtype=vtype)
    vertices['a_position'] = mesh.vertices - mesh.center_mass
    vertices['a_position'] /= np.abs(vertices['a_position']).max()
    vertices['a_position'] *= 2
    vertices['a_normal'] = mesh.vertex_normals
    # vertices['a_color'] = mesh.visual.vertex_colors / 255
    vertices['a_color'] = np.ones_like(mesh.visual.vertex_colors)
    vertices['a_color'][:,:3] *= 0.3

    filled = np.array(mesh.faces, dtype=np.uint32)

    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)

    return vertices, filled, None


def create_square():
    vtype = [('a_position', np.float32, 3), ('a_texcoord', np.float32, 2),
             ('a_normal',   np.float32, 3), ('a_color',    np.float32, 4)]
    itype = np.uint32

    # Vertices positions
    p = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
                 dtype=float)
    # Face Normals
    n = np.array([[0, 0, 1]])
    # Vertice colors
    c = np.array([[1.0, 1.0, 1.0, 1], [1.0, 1.0, 1.0, 1], [1.0, 1.0, 1.0, 1],
                  [1.0, 1.0, 1.0, 1]])
    # Texture coords
    t = np.array([[0, 0]])

    faces_p = [0, 1, 2, 3]
    faces_c = [0, 1, 2, 3]
    faces_n = [0, 0, 0, 0]
    faces_t = [0, 0, 0, 0]

    vertices = np.zeros(4, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal'] = n[faces_n]
    vertices['a_color'] = c[faces_c]
    vertices['a_texcoord'] = t[faces_t]

    filled = np.resize(
       np.array([0, 1, 2, 0, 2, 3], dtype=itype), 6)

    outline = np.resize(
        np.array([0, 1, 1, 2, 2, 3, 3, 0], dtype=itype), 6 * (2 * 4))
    outline += np.repeat(4 * np.arange(6, dtype=itype), 8)

    vertices = vertices.view(gloo.VertexBuffer)
    filled = filled.view(gloo.IndexBuffer)
    outline = outline.view(gloo.IndexBuffer)

    return vertices, filled, outline


def create_cube():
    vtype = [('a_position', np.float32, 3),
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

    faces_p = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_c = [0, 1, 2, 3,  0, 3, 4, 5,   0, 5, 6, 1,
               1, 6, 7, 2,  7, 4, 3, 2,   4, 7, 6, 5]
    faces_n = [0, 0, 0, 0,  1, 1, 1, 1,   2, 2, 2, 2,
               3, 3, 3, 3,  4, 4, 4, 4,   5, 5, 5, 5]

    vertices = np.zeros(24, vtype)
    vertices['a_position'] = p[faces_p]
    vertices['a_normal'] = n[faces_n]
    vertices['a_color'] = c[faces_c]

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


def render(shape, n_samples, imgsize, fixed_z=True,
           show=False):
    pbar = tqdm.tqdm(total=n_samples, dynamic_ncols=True)
    root = Path(shape)
    root.mkdir(exist_ok=True)
    (root / 'no_rotation').mkdir(exist_ok=True)
    (root / 'no_translation').mkdir(exist_ok=True)
    (root / 'images').mkdir(exist_ok=True)
    gt = [[n_samples, 'x', 'y', 'z', 'theta', 'phi', 'gamma']]
    x = y = z = theta = phi = gamma = 0

    with open('data.vert') as f:
        vertex = f.read()

    with open('data.frag') as f:
        fragment = f.read()

    window = app.Window(width=imgsize, height=imgsize, visible=show,
                        color=(0.0, 0.0, 0.0, 1.00))
    framebuffer = np.zeros((window.height, window.width * 3),
                           dtype=np.uint8)

    if shape == 'cube':
        V, I, _ = create_cube()
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
        cube["u_light_position"] = 3, 3, 3
        cube["u_light_intensity"] = 1, 1, 1
        cube["u_light_ambient"] = 0.2
    elif shape == 'square':
        V, I, _ = create_square()
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
        cube["u_light_position"] = 3, 3, 3
        cube["u_light_intensity"] = 0, 0, 0
        cube["u_light_ambient"] = 1
    elif shape in shapes:
        V, I, _ = load_ply(shape)
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
        cube["u_light_position"] = 3, 3, 3
        cube["u_light_intensity"] = 1, 1, 1
        cube["u_light_ambient"] = 0.2


    # cube['u_texture'] = mono()
    cube['u_model'] = np.eye(4, dtype=np.float32)
    cube['u_view'] = glm.translation(0, 0, -7)
    if shape == 'square':
        min_xy, max_xy = -2, 2
        min_z, max_z = -1, 3
    elif shape == 'cube':
        min_xy, max_xy = -1.7, 1.7
        min_z, max_z = -3, 1.8
    elif shape in shapes:
        min_xy, max_xy = -1.7, 1.7
        min_z, max_z = -3, 1.8
    frame = 0

    @window.event
    def on_draw(dt):
        nonlocal frame, x, y, z, theta, phi, gamma

        # Export screenshot
        gl.glReadPixels(0, 0, window.width,
                        window.height, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, framebuffer)
        if frame > 2:  # Skip empty zero frame
            if (frame) % 3 == 0:
                pbar.update()
                gt.append([f'{(frame-3)//3:05d}.png', x, y, z, theta, phi,
                           gamma])
                png.from_array(framebuffer,
                               'RGB').save(root / 'images' /
                                           f'{(frame-3)//3:05d}.png')
            elif (frame) % 3 == 1:
                png.from_array(framebuffer,
                               'RGB').save(root / 'no_rotation' /
                                           f'{(frame-4)//3:05d}.png')
            elif (frame) % 3 == 2:
                png.from_array(framebuffer,
                               'RGB').save(root / 'no_translation' /
                                           f'{(frame-5)//3:05d}.png')

        if (frame - 1) % 3 == 0:
            theta = np.random.random_sample() * 360
            x, y = np.random.random_sample(2) * (max_xy -
                                                 min_xy) + min_xy
            if not fixed_z:
                z = np.random.random_sample() * (max_z - min_z) + min_z
            if shape == 'cube':
                phi = np.random.random_sample() * 360
            if shape in shapes:
                phi = np.random.random_sample() * 360
                gamma = np.random.random_sample() * 360

        window.clear()

        # Fill cube
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        cube['u_color'] = 1, 1, 1, 1
        cube.draw(gl.GL_TRIANGLES, I)

        # Rotate cube
        view = cube['u_view'].reshape(4, 4)
        model = np.eye(4, dtype=np.float32)
        if (frame - 1) % 3 != 1:
            glm.rotate(model, theta, 0, 0, 1)
            glm.rotate(model, phi, 0, 1, 0)
            glm.rotate(model, gamma, 1, 0, 0)

        # Translate cube
        if (frame - 1) % 3 != 2:
            glm.translate(model, x, y, z)

        cube['u_model'] = model
        cube['u_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

        frame += 1

    @window.event
    def on_resize(width, height):
        cube['u_projection'] = glm.perspective(45.0,
                                               width / float(height),
                                               2.0, 100.0)

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPolygonOffset(1, 1)
        gl.glEnable(gl.GL_LINE_SMOOTH)

    app.run(framecount=n_samples*3 + 2, framerate=0)
    #print(gt)
    gt = np.array(gt)
    #print(gt)
    np.savetxt(root / 'target.txt', gt, delimiter='\t', fmt='%s')
    pbar.close()


def render_depth(shape, imgsize, R, t):
    root = Path(shape)
    root.mkdir(exist_ok=True)

    vertex = """
    uniform mat4   u_model;         // Model matrix
    uniform mat4   u_view;          // View matrix
    uniform mat4   u_projection;    // Projection matrix
    attribute vec4 a_color;         // Vertex color
    attribute vec3 a_position;      // Vertex position
    varying float v_eye_depth;

    void main() {
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
    vec3 v_eye_pos = (u_view * u_model * vec4(a_position, 1.0)).xyz; // Vertex position in eye coords.

    // OpenGL Z axis goes out of the screen, so depths are negative
    v_eye_depth = -v_eye_pos.z;
    }
    """

    # Depth fragment shader
    fragment = """
    varying float v_eye_depth;

    void main() {
    gl_FragColor = vec4(v_eye_depth, v_eye_depth, v_eye_depth, 1.0);
    }
    """

    window = app.Window(width=imgsize, height=imgsize, visible=False,
                        color=(0.0, 0.0, 0.0, 1.00))
    depthbuffer = np.zeros((window.height, window.width * 3),
                           dtype=np.float32)

    if shape == 'cube':
        V, I, _ = create_cube()
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
    elif shape == 'square':
        V, I, _ = create_square()
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)
    elif shape in shapes:
        V, I, _ = load_ply(shape)
        cube = gloo.Program(vertex, fragment)
        cube.bind(V)

    cube['u_model'] = np.eye(4, dtype=np.float32)
    cube['u_view'] = glm.translation(0, 0, -7)

    depth = None

    @window.event
    def on_draw(dt):
        nonlocal depth
        color_buf = np.zeros((imgsize, imgsize, 4), np.float32).view(gloo.TextureFloat2D)
        depth_buf = np.zeros((imgsize, imgsize), np.float32).view(gloo.DepthTexture)
        fbo = gloo.FrameBuffer(color=color_buf, depth=depth_buf)
        fbo.activate()

        window.clear()

        # Fill cube
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)

        # Rotate cube
        model = np.eye(4, dtype=np.float32)

        R_ = np.eye(4)
        R_[:3, :3] = R

        model = R_ @ model

        # Translate cube
        glm.translate(model, *t[0])

        cube['u_model'] = model
        # cube['u_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

        cube.draw(gl.GL_TRIANGLES, I)
        # depth = np.zeros((shape[0], shape[1], 4), dtype=np.float32)
        # gl.glReadPixels(0, 0, shape[1], shape[0], gl.GL_RGBA, gl.GL_FLOAT, depth)
        # depth.shape = shape[0], shape[1], 4
        # depth = depth[::-1, :]
        # depth = depth[:, :, 0] # Depth is saved in the first channel



        # Export screenshot
        gl.glReadPixels(0, 0, window.width,
                        window.height, gl.GL_RGB,
                        gl.GL_FLOAT, depthbuffer)
        # print(depthbuffer[depthbuffer != 0])
        # png.from_array(np.floor((depthbuffer - 0) / depthbuffer.max() * 255).astype(np.uint8),
        #                'RGB').save(root / 'images' /
        #                            'depth.png')

        fbo.deactivate()
        depth = depthbuffer.reshape((3, 128, 128))[0]

    @window.event
    def on_resize(width, height):
        cube['u_projection'] = glm.perspective(45.0,
                                               width / float(height),
                                               .1, 100.0)

    @window.event
    def on_init():
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPolygonOffset(1, 1)
        gl.glEnable(gl.GL_LINE_SMOOTH)

    app.run(framecount=1, framerate=0)
    return depth


if __name__ == '__main__':
    fire.Fire(render)
