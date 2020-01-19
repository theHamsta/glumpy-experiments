#
# Copyright © 2020 Stephan Seitz <stephan.seitz@fau.de>
#
# Distributed under terms of the GPLv3 license.

"""

"""


import numpy as np
from glumpy import app, data, gl, gloo

from pycuda.compiler import SourceModule

CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4
window = app.Window()

fragment = """
           uniform vec4 color;
           void main() {
               gl_FragColor = color;
           } """

vertex = """ uniform float time;
         attribute vec2 position;
         void main()
         {
             vec2 xy = vec2(sin(2.0*time));
             gl_Position = vec4(position*(0.25 + 0.75*xy*xy), 0.0, 1.0);
         } """
quad = gloo.Program(vertex, fragment, count=4)

GridHeight =  400
GridWidth =  400


class Surface(object):
    def __init__(self, width, height, depth, interpolation=gl.GL_NEAREST):
        self.texture = np.zeros((height,width,depth), np.float32).view(gloo.TextureFloat2D)
        self.texture.interpolation = interpolation
        self.framebuffer = gloo.FrameBuffer(color=self.texture)
        self.clear()

    def clear(self):
        self.activate()
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        self.deactivate()

    def activate(self):
        self.framebuffer.activate()

    def deactivate(self):
        self.framebuffer.deactivate()

class Slab(object):
    def __init__(self, width, height, depth, interpolation=gl.GL_NEAREST):
        import pycuda.gl.autoinit
        self.Ping = Surface(width, height, depth, interpolation)
        self.Pong = Surface(width, height, depth, interpolation)
        self._ping_cuda = pycuda.gl.RegisteredImage(self.Ping.texture._handle,
                                                    gl.GL_TEXTURE_2D).map()
        self._pong_cuda = pycuda.gl.RegisteredImage(self.Pong.texture._handle,
                                                    gl.GL_TEXTURE_2D).map()

    def swap(self):
        self.Ping, self.Pong = self.Pong, self.Ping
        self._ping_cuda, self._pong_cuda = self._pong_cuda, self._ping_cuda

    @property
    def ping_array(self):
        return self._ping_cuda.array(0,0)

    @property
    def pong_array(self):
        return self._pong_cuda.array(0,0)

Density = Slab(GridWidth, GridHeight, 1, gl.GL_LINEAR)

mod = SourceModule("""
surface<void, 2> surf;
__global__ void kernel(int width, int height)
{
    // Calculate surface coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 400 && y < 400) {
        float data = x / 400.f;
        // Write to output surface
        surf2Dwrite(data, surf, x*4, y);
    }
}
""")

kernel_function = mod.get_function('kernel')
surface_ref = mod.get_surfref('surf')
# surface_ref.set_array(Density.ping_array,0)
surface_ref.set_array(Density.ping_array)

def Program(fragment):
    program = gloo.Program("vertex_passthrough.vert", fragment, count=4)
    program['Position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
    return program



Density = Slab(GridWidth, GridHeight, 1, gl.GL_LINEAR)

prog_visualize = Program("visualize.frag")

def ClearSurface(surface, v):
    surface.activate()
    gl.glClearColor(v, v, v, v)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    surface.deactivate()

@window.event
def on_init():
    gl.glDisable(gl.GL_DEPTH_TEST)
    gl.glDisable(gl.GL_BLEND)

t = 0
@window.event
def on_draw(dt):

    surface_ref.set_array(Density.ping_array)
    kernel_function(np.int32(400), np.int32(400), block=(16,16,1), grid=((400+1)//16+1,(400+1)//16+1))

    gl.glViewport(0,0,window.width,window.height)
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    global t
    t += dt
    prog_visualize['u_data']   = Density.Ping.texture
    prog_visualize['t'] = t
    prog_visualize['u_shape']  = Density.Ping.texture.shape[1], Density.Ping.texture.shape[0]
    prog_visualize['u_kernel'] = data.get("spatial-filters.npy")
    prog_visualize["Sampler"] = Density.Ping.texture
    prog_visualize["FillColor"] = 0.95, 0.925, 1.00
    prog_visualize["Scale"] =  1.0/window.width, 1.0/window.height
    prog_visualize.draw(gl.GL_TRIANGLE_STRIP)
    Density.swap()

app.run()
