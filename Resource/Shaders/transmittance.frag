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
	float reflectFactor;
	float refractiveIndex;
};
                                                               
uniform MaterialInfo material;

void main()
{ 
	// calculate intermediary values
	vec3 normal = normalize (Normal0);
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
