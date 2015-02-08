#version 430

const float PI = 3.14159;
const int MAX_DIRENCTIONAL_LIGHTS = 2;
const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

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

// Material properties
struct MaterialInfo
{
    vec4 Ka; // Ambient reflectivity
    vec4 Kd; // Diffuse reflectivity
    vec4 Ks; // Specular reflectivity
    vec4 Ke; // Emissive reflectivity

	bool hasDiffuseMap;
	bool hasNormalMap;

	float shininessStrength; // Specular intensity
    float shininess; // Specular shininess exponent

	// below is used for cook torrance shader
	float roughnessValue; // 0 : smooth, 1: rough
	float fresnelReflectance; // fresnel reflectance at normal incidence
};


struct BaseLight
{
    vec4 Color;
    float Intensity;
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
    vec3 Direction;                                                                         
    float Cutoff;                                                                           
};                                                                                          

uniform int gNumDirectionalLights;                                                                                            
uniform int gNumPointLights;                                                                
uniform int gNumSpotLights;
uniform DirectionalLight gDirectionalLights[MAX_DIRENCTIONAL_LIGHTS]; 
uniform PointLight gPointLights[MAX_POINT_LIGHTS];
uniform SpotLight gSpotLights[MAX_SPOT_LIGHTS];
uniform sampler2D gColorMap;                                                                
uniform sampler2D gShadowMap;                                                         
uniform sampler2D gNormalMap;                                                              
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
    vec4 AmbientColor = material.Ka * Light.Color * Light.Intensity;                   
    vec4 EmissiveColor = material.Ke * Light.Color;     
	                                                                                                                                                                    
    vec4 DiffuseColor  = vec4(0, 0, 0, 0);                                                  
    vec4 SpecularColor = vec4(0, 0, 0, 0);          
	vec4 FinalColor = vec4(0, 0, 0, 0);
	                    
    float DiffuseFactor = dot(In.Normal, -LightDirection);
    if (DiffuseFactor > 0.0) 
	{                                                         
        DiffuseColor = material.Kd * Light.Color * Light.Intensity * DiffuseFactor;    
                                                                                            
        vec3 VertexToEye = normalize(gEyeWorldPos - In.WorldPos);                             
		float OutlineFactor = dot(VertexToEye, In.Normal);
		
        SpecularColor = material.Ks * Light.Color;                         
   
		if(DiffuseFactor < 0.2)
		{
			FinalColor = AmbientColor;
		}
		else if(DiffuseFactor >= 0.2 && DiffuseFactor < 0.8)
		{
			FinalColor = DiffuseColor;
		}
		else
		{
			FinalColor = SpecularColor;
		} 
		if(OutlineFactor < 0.3)
		{
			FinalColor = vec4(0, 0, 0, 0);
		}                                                                        
    }                                                                                       
                                                                                          
    return (EmissiveColor + ShadowFactor * FinalColor); 
}                                                                                           
                                                                                            
vec4 CalcDirectionalLight(DirectionalLight l, VSOutput In)                                                      
{                                                                                           
    return CalcLightInternal(l.Base, l.Direction, In, 1.0);  
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
    vec3 LightToPixel = normalize(In.WorldPos - l.Base.Position);                             
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
    In.Color = Color0;
    In.TexCoord = TexCoord0;
	In.Normal   = normalize(Normal0);
    In.WorldPos = WorldPos0;
	In.Tangent = Tangent0;

    vec4 TotalLight;
	
	for (int i = 0 ; i < gNumDirectionalLights ; i++) {                                           
        TotalLight += CalcDirectionalLight(gDirectionalLights[i], In);                              
    } 
                                                                                            
    for (int i = 0 ; i < gNumPointLights ; i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], In);                              
    }                                                                                       
                                                                                            
    for (int i = 0 ; i < gNumSpotLights ; i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], In);                                
    }                                                                                       
    
	vec4 baseColor;
	if(material.hasDiffuseMap)
		baseColor = texture(gColorMap, In.TexCoord.xy);
	else
		baseColor = Color0;                                                      
                                                                                            
    FragColor = baseColor * TotalLight;     
}
