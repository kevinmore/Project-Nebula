#version 430 core

/*
PER-FRAGMENT LIGHTING (PHONG SHADING IMPLEMENTATION)
*/

in vec3 position; // Vertex position
in vec4 color;    // Vertex color
in vec2 texCoord; // Texture coordinates
in vec3 normal;   // Vertex normal
in ivec4 BoneIDs; // Vertex Bone IDs
in vec4 Weights;  // Vertex Bone Weights

out VS_OUT
{
    vec4 P; // Position vector transformed into eye (camera) space
    vec3 N; // Normal vector transformed into eye (camera) space
    vec3 V; // View vector (the negative of the view-space position)

    vec2 texCoord;
    vec4 color;
} vs_out;

uniform mat3 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

const int MAX_BONES = 100;
uniform mat4 gBones[MAX_BONES];

void main()
{
	mat4 modelViewMatrix = viewMatrix * modelMatrix;

	// Calculate the bone transform matrix
	mat4 BoneTransform = gBones[BoneIDs[0]] * Weights[0];
    BoneTransform     += gBones[BoneIDs[1]] * Weights[1];
    BoneTransform     += gBones[BoneIDs[2]] * Weights[2];
    BoneTransform     += gBones[BoneIDs[3]] * Weights[3];

	vec4 PosL  = BoneTransform * vec4(position, 1.0);

    // Calculate position vector in view-space coordinate
    //vs_out.P = modelViewMatrix * vec4(position, 1.0);
	vs_out.P = modelViewMatrix * PosL;



    // Calculate normal vector in view-space coordinate
    // normalMatrix is the transpose of the inverse of the modelView matrix
    // normalMatrix = transpose(inverse(mat3(modelViewMatrix)));
    //vs_out.N = normalMatrix * normal;

 	vec3 NormalL = (BoneTransform * vec4(normal, 0.0)).xyz;
 	vs_out.N = normalMatrix * NormalL;


    // Calculate viewing vector from eye position to surface point
    vs_out.V = -vs_out.P.xyz;

    // Send texture coordinates and vertex color to the fragment shader
    vs_out.texCoord = texCoord;
    vs_out.color    = color;

    // Calculate the clip-space position of each vertex
    gl_Position = projectionMatrix * vs_out.P;
}
