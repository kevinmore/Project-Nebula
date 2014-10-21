#version 330
const int MAX_BONES = 100;

uniform mat4 mvpMatrix;
uniform mat4 mvMatrix;
uniform mat3 normalMatrix;
uniform vec3 lightPosition;



in vec4 vertex;
in vec4 color;
in vec3 normal;
in vec2 textureCoordinate;
in ivec4 BoneIDs;
in vec4 Weights;

out vec3 varyingNormal;
out vec3 varyingLightDirection;
out vec3 varyingViewerDirection;
out vec2 varyingTextureCoordinate;
out vec4 varyingColor;

uniform mat4 gBones[MAX_BONES];

void main(void)
{
	mat4 BoneTransform = gBones[BoneIDs[0]] * Weights[0];
    BoneTransform += gBones[BoneIDs[1]] * Weights[1];
    BoneTransform += gBones[BoneIDs[2]] * Weights[2];
    BoneTransform += gBones[BoneIDs[3]] * Weights[3];

	vec4 PosL = BoneTransform * vertex;
	gl_Position = mvpMatrix * PosL;

	varyingTextureCoordinate = textureCoordinate;

	vec4 NormalL = BoneTransform * vec4(normal, 0.0);
	varyingNormal =  (mvMatrix * NormalL).xyz;


    vec4 eyeVertex = mvMatrix * vertex;
    eyeVertex /= eyeVertex.w;


    varyingLightDirection = lightPosition - eyeVertex.xyz;
    varyingViewerDirection = -eyeVertex.xyz;
    
    
	varyingColor = color;
}
