#version 430
layout (location = 0) in float nodeMass;
layout (location = 1) in vec3 nodeVelocity;
layout (location = 2) in vec3 nodeForce;

out vec4 nodeColor;

uniform vec3 pos;
uniform vec3 dim;
uniform float h;
uniform float density;
uniform mat4 gWVP;
uniform int mode;

const int MASS = 0;
const int VELOCITY = 1;
const int SPEED = 2;
const int FORCE = 3;

void main( void )
{
    float alpha = 0.75 * smoothstep( 0.0, density, nodeMass/(h*h*h) );
    nodeColor = vec4( 0.8, 0.8, 0.9, alpha );

    float i = gl_VertexID;
    float x = floor(i/((dim.y+1)*(dim.z+1)));
    i -= x*(dim.y+1)*(dim.z+1);
    float y = floor(i/(dim.z+1));
    i -= y*(dim.z+1);
    float z = i;
    vec4 position = vec4( pos + h * vec3( x, y, z ), 1.0 );
    gl_Position = gWVP * position;
    gl_PointSize = 3.0;
}
