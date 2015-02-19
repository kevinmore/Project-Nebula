#version 430

in vec3 Normal0;                                                       
in vec3 WorldPos0;                                                           

out vec4 FragColor;

uniform samplerCube gCubemapTexture; 
uniform vec3 gEyeWorldPos;

// Material properties
struct MaterialInfo
{
	// below is used for transmittance
	float fresnelReflectance; // fresnel reflectance at normal incidence
	float refractiveIndex;
};
                                                               
uniform MaterialInfo material;

void main()
{ 
	// Compute the refracted direction in world coords.
	vec3 normal = normalize (Normal0);
	vec3 eyeDir = normalize(WorldPos0 - gEyeWorldPos);
	vec3 reflectDir = reflect(eyeDir, normal);
	vec3 refractDir = refract(eyeDir, normal, 1.0 / material.refractiveIndex);                              
	
	// Access the cube map texture
	vec4 reflectColor = texture(gCubemapTexture, reflectDir);
	vec4 refractColor = texture(gCubemapTexture, refractDir);


	FragColor = mix(refractColor, reflectColor, material.fresnelReflectance);
}
