#version 430                                                                        
                                                                                    
layout (location = 0) in vec3 Position;                                             
                                  
uniform mat4 gWVP;                                                                                                                        
                                                                                                                       

void main()
{   
    vec4 PosL    = vec4(Position, 1.0);

    gl_Position  = gWVP * PosL;             
}
