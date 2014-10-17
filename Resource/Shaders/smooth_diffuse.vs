#version 330

// Note: Input to this shader is the vertex positions that we specified for the triangle. 
// Note: gl_Position is a special built-in variable that is supposed to contain the vertex position (in X, Y, Z, W)
// Since our triangle vertices were specified as vec3, we just set W to 1.0.

in vec3 vPosition;
in vec4 vColor;
in vec3 vNormal;

// model view projection matrix
uniform mat4 MVP;
uniform mat4 modelToWorld;

// light
uniform vec3 ambientLight;
uniform vec3 lightPosition;

out vec3 position;
out vec3 normal;
out vec4 color;

void main()
{
	gl_Position = MVP * vec4(vPosition, 1.0);
	normal = vNormal;
	vec4 translatedPos = modelToWorld * vec4(vPosition, 1.0);
	position = vec3(translatedPos.x, translatedPos.y, translatedPos.z);
	color = vColor;
}