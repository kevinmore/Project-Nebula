#version 430

const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

in vec4 LightSpacePos0; 
in vec2 TexCoord0;
in vec3 Normal0;                                                       
in vec3 WorldPos0;                                                           

out vec4 FragColor;

// Material properties
struct MaterialInfo
{
    vec4 Ka; // Ambient reflectivity
    vec4 Kd; // Diffuse reflectivity
    vec4 Ks; // Specular reflectivity
    vec4 Ke; // Emissive reflectivity

	// below is used for phong shader
	float shininessStrength; // Specular intensity
    float shininess; // Specular shininess exponent

	// below is used for cook torrance shader
	float roughnessValue; // 0 : smooth, 1: rough
	float fresnelReflectance; // fresnel reflectance at normal incidence
};

struct VSOutput
{
    vec2 TexCoord;
    vec3 Normal;                                                                   
    vec3 WorldPos;
	vec4 LightSpacePos;                                                                 
};


struct BaseLight
{
    vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;
};

struct DirectionalLight
{
    BaseLight Base;
    vec3 Direction;
};
                                                                                    
struct Attenuation                                                                  
{                                                                                   
    float Constant;                                                                 
    float Linear;                                                                   
    float Exp;                                                                      
};                                                                                  
                                                                                    
struct PointLight                                                                           
{                                                                                           
    BaseLight Base;                                                                  
    vec3 Position;                                                                          
    Attenuation Atten;                                                                      
};                                                                                          
                                                                                            
struct SpotLight                                                                            
{                                                                                           
    PointLight Base;  
	vec3 Position;                                                               
    vec3 Direction;                                                                         
    float Cutoff;                                                                           
};                                                                                          
                                                                                            
uniform int gNumPointLights;                                                                
uniform int gNumSpotLights;                                                                 
uniform DirectionalLight gDirectionalLight;                                                 
uniform PointLight gPointLights[MAX_POINT_LIGHTS];                                          
uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];                                             
uniform sampler2D gColorMap;                                                                
uniform sampler2D gShadowMap;                                                         
uniform vec3 gEyeWorldPos;                                                                  
uniform MaterialInfo material;

float CalcShadowFactor(vec4 LightSpacePos)                                                  
{                                                                                           
    vec3 ProjCoords = LightSpacePos.xyz / LightSpacePos.w;                                  
    vec2 UVCoords;                                                                          
    UVCoords.x = 0.5 * ProjCoords.x + 0.5;                                                  
    UVCoords.y = 0.5 * ProjCoords.y + 0.5;                                                  
    float Depth = texture(gShadowMap, UVCoords).x;                                       
    if (Depth <= (ProjCoords.z + 0.005))                                                    
        return 0.5;                                                                         
    else                                                                                    
        return 1.0;                                                                         
} 

vec4 CalcLightInternal(BaseLight Light, vec3 LightDirection, VSOutput In, float ShadowFactor)            
{   
	// Compute the ambient / diffuse / specular / emissive components for each fragment
    vec4 AmbientColor = material.Ka * vec4(Light.Color, 1.0) * Light.AmbientIntensity;                   
    vec4 EmissiveColor = material.Ke;     
	vec4 DiffuseColor;                                                  
    vec4 SpecularColor;
	                                                                                                                                                                                                                                                             
    float NdotL = max(dot(In.Normal, -LightDirection), 0.0);
    float specular = 0.0;
	DiffuseColor = material.Kd * vec4(Light.Color, 1.0) * Light.DiffuseIntensity * NdotL;
    if (NdotL > 0.0) 
	{   
		                                                               
        vec3 eyeDir  = normalize(gEyeWorldPos - In.WorldPos);                             
        // calculate intermediary values
        vec3 halfVector = normalize( eyeDir - LightDirection);
        float NdotH = max(dot(In.Normal, halfVector), 0.0); 
        float NdotV = max(dot(In.Normal, eyeDir), 0.0); // note: this could also be NdotL, which is the same value
        float VdotH = max(dot(eyeDir, halfVector), 0.0);
        float mSquared = material.roughnessValue  * material.roughnessValue ;     
		
		// geometric attenuation
        float NH2 = 2.0 * NdotH;
        float g1 = (NH2 * NdotV) / VdotH;
        float g2 = (NH2 * NdotL) / VdotH;
        float geoAtt = min(1.0, min(g1, g2));
		
		 // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * mSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (mSquared * NdotH * NdotH);
        float roughness = r1 * exp(r2);
		
		// fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - material.fresnelReflectance);
        fresnel += material.fresnelReflectance;              
		
		specular = (fresnel * geoAtt * roughness) / (NdotV * NdotL * 3.14);                                                               
    }                                                                                       
	vec3 finalValue = Light.Color * NdotL * specular * material.shininessStrength;
	SpecularColor = material.Ks * vec4(finalValue, 1.0);                                                                                         
    return (AmbientColor + EmissiveColor + ShadowFactor * (DiffuseColor + SpecularColor)); 
}                                                                                           
                                                                                            
vec4 CalcDirectionalLight(VSOutput In)                                                      
{                                                                                           
    return CalcLightInternal(gDirectionalLight.Base, gDirectionalLight.Direction, In, 1.0);  
}                                                                                           
                                                                                            
vec4 CalcPointLight(PointLight l, VSOutput In)                                       
{                                                                                           
    vec3 LightDirection = In.WorldPos - l.Position;                                           
    float Distance = length(LightDirection);                                                
    LightDirection = normalize(LightDirection);                                             
    
	float ShadowFactor = 1;//CalcShadowFactor(In.LightSpacePos);
	                                                                               
    vec4 Color = CalcLightInternal(l.Base, LightDirection, In, 1);                         
    float Attenuation =  l.Atten.Constant +                                                 
                         l.Atten.Linear * Distance +                                        
                         l.Atten.Exp * Distance * Distance;                                 
                                                                                            
    return Color / Attenuation;                                                             
}                                                                                           
                                                                                            
vec4 CalcSpotLight(SpotLight l, VSOutput In)                                         
{                                                                                           
    vec3 LightToPixel = normalize(In.WorldPos - l.Position);                             
    float SpotFactor = dot(LightToPixel, l.Direction);                                      
                                                                                            
    if (SpotFactor > l.Cutoff) {                                                            
        vec4 Color = CalcPointLight(l.Base, In);                                        
        return Color * (1.0 - (1.0 - SpotFactor) * 1.0/(1.0 - l.Cutoff));                   
    }                                                                                       
    else {                                                                                  
        return vec4(0,0,0,0);                                                               
    }                                                                                       
}                                                                                           
                             
                                                                
void main()
{                                    
    VSOutput In;
    In.TexCoord = TexCoord0;
	In.Normal   = normalize(Normal0);
    In.WorldPos = WorldPos0;
	In.LightSpacePos = LightSpacePos0;

	

    vec4 TotalLight = CalcDirectionalLight(In);                                         
                                                                                            
    for (int i = 0 ; i < gNumPointLights ; i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], In);                              
    }                                                                                       
    
	TotalLight /= 2.0; 
	                                                                                       
    for (int i = 0 ; i < gNumSpotLights ; i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], In);                                
    }                                                                                       
                                                                                            
    FragColor = texture(gColorMap, In.TexCoord.xy) * TotalLight;     
}
