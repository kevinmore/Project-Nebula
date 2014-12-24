#version 430                                                                        
                                                                                    
layout (location = 0) in vec3 Position;                                             
layout (location = 1) in vec2 TexCoord;                                             
layout (location = 2) in vec3 Normal;                                               
layout (location = 3) in vec3 Tangent;

uniform mat4 gWVP;                                                                  
uniform mat4 gLightWVP;                                                             
uniform mat4 gWorld;                                                                
                                                                                    
out vec4 LightSpacePos0;                                                             
out vec2 TexCoord0;                                                                 
out vec3 Normal0;                                                                   
out vec3 WorldPos0;                                                                 
out vec3 Tangent0; 

void main()
{   
    vec4 PosL    = vec4(Position, 1.0);

    gl_Position  = gWVP * PosL;
	LightSpacePos0 = gLightWVP * PosL;
    TexCoord0    = TexCoord;
    vec4 NormalL = vec4(Normal, 0.0);
    Normal0      = (gWorld * NormalL).xyz;
    WorldPos0    = (gWorld * PosL).xyz;                
	Tangent0     = (gWorld * vec4(Tangent, 0.0)).xyz;                   
}
