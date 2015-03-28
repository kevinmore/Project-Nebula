#version 430

layout (location = 0) in vec3 particlePosition;
layout (location = 1) in vec3 particleVelocity;
layout (location = 2) in float particleMass;
layout (location = 3) in float particleVolume;
layout (location = 4) in float particleStiffness; // mat.xi

uniform int mode;
uniform mat4 gWVP;                                                                  

const int MASS = 0;
const int VELOCITY = 1;
const int SPEED = 2;
const int STIFFNESS = 3;

void main( void )
{
    gl_Position = gWVP * vec4( particlePosition, 1.0 );
    gl_PointSize = 3.0;
}
