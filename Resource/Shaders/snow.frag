#version 430

in vec4 particleColor;

out vec4 fragmentColor;

void main( void )
{
    fragmentColor = particleColor;
}
