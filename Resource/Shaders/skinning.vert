#version 430                                                                        
                                                                                    
layout (location = 0) in vec3 Position;                                             
layout (location = 1) in vec2 TexCoord;                                             
layout (location = 2) in vec3 Normal;
layout (location = 3) in vec3 Tangent;                                          
layout (location = 4) in vec4 Color;                                             
layout (location = 5) in ivec4 BoneIDs;
layout (location = 6) in vec4 Weights;

const int MAX_BONES = 200;

uniform mat4 gWVP;                                                                  
uniform mat4 gLightWVP;                                                             
uniform mat4 gWorld;  
uniform mat4 gBones[MAX_BONES];

out vec4 LightSpacePos0;                                                             
out vec2 TexCoord0;
out vec4 Color0;                                                                 
out vec3 Normal0;                                                                   
out vec3 WorldPos0;                                                                 
out vec3 Tangent0; 


void main()
{   
	mat4 BoneTransform;

	BoneTransform = gBones[BoneIDs[0]] * Weights[0];
    BoneTransform += gBones[BoneIDs[1]] * Weights[1];
    BoneTransform += gBones[BoneIDs[2]] * Weights[2];
    BoneTransform += gBones[BoneIDs[3]] * Weights[3];

	
    vec4 PosL    = BoneTransform * vec4(Position, 1.0);
	LightSpacePos0 = gLightWVP * PosL;
    gl_Position  = gWVP * PosL;
	Color0 = Color;
    TexCoord0    = TexCoord;
    vec4 NormalL = BoneTransform * vec4(Normal, 0.0);
    Normal0      = (gWorld * NormalL).xyz;
    WorldPos0    = (gWorld * PosL).xyz;                                
    vec4 TangentL = BoneTransform * vec4(Tangent, 0.0);
	Tangent0     = (gWorld * TangentL).xyz;                   
}
