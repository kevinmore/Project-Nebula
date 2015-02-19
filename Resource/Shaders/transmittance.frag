#version 430

in vec2 TexCoord0;                                                                 
in vec4 Color0;                                                                 
in vec3 Normal0;                                                                   
in vec3 WorldPos0;                                                                 
in vec3 Tangent0;                                                          

out vec4 FragColor;

uniform samplerCube gCubemapTexture;
uniform sampler2D gNormalMap;
uniform vec3 gEyeWorldPos;

// Material properties
struct MaterialInfo
{
	bool hasNormalMap;

	// below is used for transmittance
	float reflectFactor;
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
	// calculate intermediary values
	vec3 normal;
	if(material.hasNormalMap)
		normal = CalcBumpedNormal();
	else
		normal = normalize (Normal0);

	vec3 eyeDir = normalize(gEyeWorldPos - WorldPos0);
    float VdotN = dot(eyeDir, normal);

	float EtaG = 1.0 / material.refractiveIndex; // Ratio of indices of refraction
	float EtaR = EtaG - 0.2;
	float EtaB = EtaG + 0.2;

	float F = ((1.0-EtaG) * (1.0-EtaG)) / ((1.0+EtaG) * (1.0+EtaG));
	// fresnel
    // Schlick approximation
	float fresnel = F + (1.0 - F) * pow(1.0 - VdotN, 5.0);    

	// Compute the refracted direction in world coords.
	vec3 refractDirR = refract(eyeDir, normal, EtaR);                              
	vec3 refractDirG = refract(eyeDir, normal, EtaG);                              
	vec3 refractDirB = refract(eyeDir, normal, EtaB);                              
	vec3 reflectDir = reflect(eyeDir, normal);
	
	// Access the cube map texture
	float refractColorR = texture(gCubemapTexture, refractDirR).r;
	float refractColorG = texture(gCubemapTexture, refractDirG).g;
	float refractColorB = texture(gCubemapTexture, refractDirB).b;
	vec4 refractColor = vec4(refractColorR, refractColorG, refractColorB, 1.0);

	vec4 reflectColor = texture(gCubemapTexture, reflectDir);

	FragColor = mix(refractColor, reflectColor, fresnel);
}
