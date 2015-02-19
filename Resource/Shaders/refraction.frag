#version 430

in vec2 TexCoord0;                                                                 
in vec4 Color0;                                                                 
in vec3 Normal0;                                                                   
in vec3 WorldPos0;                                                                 
in vec3 Tangent0; 
                                                        

out vec4 FragColor;

struct VSOutput
{
    vec4 Color;
	vec2 TexCoord;                                                                 
    vec3 Normal;
	vec3 Tangent;                                                                   
    vec3 WorldPos;
};

uniform sampler2D gNormalMap;   
uniform samplerCube gCubemapTexture; 
uniform vec3 gEyeWorldPos;

// Material properties
struct MaterialInfo
{
	bool hasNormalMap;
	// below is used for transmittance
	float fresnelReflectance; // fresnel reflectance at normal incidence
	float refractiveIndex;
};
                                                               
uniform MaterialInfo material;

vec3 CalcBumpedNormal()                                                                     
{                                                                                           
    vec3 Normal = normalize(Normal0);                                                       
    vec3 Tangent = normalize(Tangent0);                                                     
    Tangent = normalize(Tangent - dot(Tangent, Normal) * Normal);                           
    vec3 Bitangent = cross(Tangent, Normal);                                                
    vec3 BumpMapNormal = texture(gNormalMap, TexCoord0).xyz;                                
    BumpMapNormal = 2.0 * BumpMapNormal - vec3(1.0, 1.0, 1.0);                              
    vec3 NewNormal;                                                                         
    mat3 TBN = mat3(Tangent, Bitangent, Normal);                                            
    NewNormal = TBN * BumpMapNormal;                                                        
    NewNormal = normalize(NewNormal);                                                       
    return NewNormal;                                                                       
} 

void main()
{ 
	VSOutput In;
	In.Color = Color0;
    In.TexCoord = TexCoord0;
    In.WorldPos = WorldPos0;
	In.Tangent = Tangent0;

	if(material.hasNormalMap)
		In.Normal   = CalcBumpedNormal();
	else
		In.Normal   = normalize(Normal0);

	// Compute the refracted direction in world coords.
	vec3 eyeDir = normalize(WorldPos0 - gEyeWorldPos);
	vec3 reflectDir = reflect(eyeDir, In.Normal);
	vec3 refractDir = refract(eyeDir, In.Normal, 1.0 / material.refractiveIndex);                              
	
	// Access the cube map texture
	vec4 reflectColor = texture(gCubemapTexture, reflectDir);
	vec4 refractColor = texture(gCubemapTexture, refractDir);


	FragColor = mix(refractColor, reflectColor, material.fresnelReflectance);
}
