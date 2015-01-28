#version 430
/*
PER-FRAGMENT LIGHTING (PHONG SHADING IMPLEMENTATION)
*/

layout (location = 0) in vec3 Position;  // Vertex position                                           
layout (location = 1) in vec4 Color;     // Vertex color                                         
layout (location = 2) in vec3 Normal;    // Vertex normal                            
layout (location = 3) in vec3 Tangent;   // Vertex tangent

out VS_OUT
{
    vec4 P; // Position vector transformed into eye (camera) space
    vec3 N; // Normal vector transformed into eye (camera) space
    vec4 color;
} vs_out;


uniform mat4 ModelMatrix;
uniform mat4 MVPMatrix;
uniform vec4 vColor;         // user definded color
uniform int customizedColor; // 0 means use color vbo, 1 means use uniform color

void main()
{
	 // Calculate the clip-space position of each vertex
	vec4 PosL = vec4(Position, 1.0);
    gl_Position = MVPMatrix * PosL;

    // Calculate position vector in view-space coordinate
    vs_out.P = (ModelMatrix * PosL).xyz;

    // Calculate normal vector in view-space coordinate
	vec4 NormalL = vec4(Normal, 0.0);
    vs_out.N = ModelMatrix * NormalL;

	// Check if the user has specified vertex color
	if(customizedColor == 0)
		vs_out.color = Color;
	else if(customizedColor == 1)
		vs_out.color = vColor;
}
