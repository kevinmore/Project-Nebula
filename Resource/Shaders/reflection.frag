#version 430

in vec3 Normal0;                                                       
in vec3 WorldPos0;                                                           

out vec4 FragColor;

uniform mat4 viewMatrix; // view matrix                                               
uniform samplerCube gCubemapTexture; 
                                                                
void main()
{                                    
  /* reflect ray around normal from eye to surface */
  vec3 incident_eye = normalize (WorldPos0);
  vec3 normal = normalize (Normal0);

  vec3 reflected = reflect (incident_eye, normal);
  // convert from eye to world space
  reflected = vec3 (inverse (viewMatrix) * vec4 (reflected, 0.0));

  FragColor = texture (gCubemapTexture, reflected);   
}
