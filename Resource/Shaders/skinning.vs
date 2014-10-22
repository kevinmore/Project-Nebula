#version 330                                                                        
                                                                                    
in vec3 Position;                                             
in vec2 TexCoord;                                             
in vec3 Normal;                                               
in ivec4 BoneIDs;
in vec4 Weights;


out vec2 TexCoord0;
out vec3 Normal0;                                                                   
out vec3 WorldPos0;                                                                 

const int MAX_BONES = 200;

uniform mat4 gWVP;
uniform mat4 gWorld;
uniform mat4 gBones[MAX_BONES];

void main()
{       
    mat4 BoneTransform = gBones[BoneIDs[0]] * Weights[0];
    BoneTransform     += gBones[BoneIDs[1]] * Weights[1];
    BoneTransform     += gBones[BoneIDs[2]] * Weights[2];
    BoneTransform     += gBones[BoneIDs[3]] * Weights[3];
	
    vec4 PosL    = BoneTransform * vec4(Position, 1.0);
     gl_Position  = gWVP * PosL;
   vec4 NormalL = BoneTransform * vec4(Normal, 0.0);
    Normal0      = (gWorld * NormalL).xyz;
     WorldPos0    = (gWorld * PosL).xyz;    
                            
}
