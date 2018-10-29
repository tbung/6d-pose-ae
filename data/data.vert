uniform mat4   u_model;         // Model matrix
uniform mat4   u_view;          // View matrix
uniform mat4   u_projection;    // Projection matrix
attribute vec4 a_color;         // Vertex color
attribute vec3 a_position;      // Vertex position
attribute vec3 a_normal;        // Vertex normal
attribute vec2 a_texcoord;      // Vertex texture coordinates
varying vec4   v_color;         // Interpolated fragment color (out)
varying vec3   v_normal;        // Interpolated normal (out)
varying vec3   v_position;      // Interpolated position (out)
varying vec2   v_texcoord;      // Interpolated fragment texture coordinates (out)

void main()
{
    // Assign varying variables
    v_color    = a_color;
    v_normal   = a_normal;
    v_position = a_position;
    v_texcoord = a_texcoord;

    // Final position
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
