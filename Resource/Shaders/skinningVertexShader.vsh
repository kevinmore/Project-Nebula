#version 330
const int MAX_BONES = 200;

uniform mat4 mMatrix;
uniform mat4 mvpMatrix;
uniform mat4 mvMatrix;
uniform mat3 normalMatrix;
uniform vec3 lightPosition;

in vec3 Position; 
in vec2 TexCoord; 
in vec4 color;
in vec3 Normal;   
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

	mat4 BoneTransform = gBones[BoneIDs.x] * Weights.x;
    BoneTransform	  += gBones[BoneIDs.y] * Weights.y;
    BoneTransform	  += gBones[BoneIDs.z] * Weights.z;
    BoneTransform	  += gBones[BoneIDs.w] * Weights.w;

	vec4 PosL = BoneTransform * vec4(Position, 1.0);
	//gl_Position = mvpMatrix * PosL;
	gl_Position = mvpMatrix * vec4(Position, 1.0);
	vec4 NormalL = BoneTransform * vec4(Normal, 0.0);
	varyingNormal =  (mvMatrix * vec4(Normal, 0.0)).xyz;


    //vec4 eyeVertex = mvMatrix * PosL;
	vec4 eyeVertex = mvMatrix * vec4(Position, 1.0);
    eyeVertex /= eyeVertex.w;


    varyingLightDirection = lightPosition - eyeVertex.xyz;
    varyingViewerDirection = -eyeVertex.xyz;
    
    varyingTextureCoordinate = TexCoord;
	varyingColor = color;
}
