uniform mat4      u_model;           // Model matrix
uniform mat4      u_view;            // View matrix
uniform mat4      u_normal;          // Normal matrix
uniform mat4      u_projection;      // Projection matrix
uniform vec4      u_color;           // Global color
/* uniform sampler2D u_texture;         // Texture */ 
uniform vec3      u_light_position;  // Light position
uniform vec3      u_light_intensity; // Light intensity
uniform float     u_light_ambient;   // Light intensity

varying vec4      v_color;           // Interpolated fragment color (in)
varying vec3      v_normal;          // Interpolated normal (in)
varying vec3      v_position;        // Interpolated position (in)
/* varying vec2      v_texcoord;        // Interpolated fragment texture coordinates (in) */
void main()
{
    // Calculate normal in world coordinates
    vec3 normal = normalize(u_normal * vec4(v_normal,1.0)).xyz;

    // Calculate the location of this fragment (pixel) in world coordinates
    vec3 position = vec3(u_view*u_model * vec4(v_position, 1));

    // Calculate the vector from this pixels surface to the light source
    vec3 surfaceToLight = u_light_position - position;

    // Calculate the cosine of the angle of incidence (brightness)
    float brightness = dot(normal, surfaceToLight) /
                      (length(surfaceToLight) * length(normal));
    brightness = max(min(brightness,1.0),0.0);

    // Calculate final color of the pixel, based on:
    // 1. The angle of incidence: brightness
    // 2. The color/intensities of the light: light.intensities
    // 3. The texture and texture coord: texture(tex, fragTexCoord)

    // Get texture color
    /* vec4 t_color = vec4(vec3(texture2D(u_texture, v_texcoord).r), 1.0); */

    // Final color
    /* vec4 color = u_color * t_color * mix(v_color, t_color, 0.25); */
    vec4 color = u_color * v_color;

    gl_FragColor = color * (brightness * vec4(u_light_intensity, 1) + u_light_ambient);
}
