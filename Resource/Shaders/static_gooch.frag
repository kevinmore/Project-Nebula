#version 430

const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

in vec4 LightSpacePos0; 
in vec4 Color0;
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

	float shininessStrength; // Specular intensity
    float shininess; // Specular shininess exponent
};

struct VSOutput
{
    vec4 Color;
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
uniform sampler2D gShadowMap;                                                         
uniform vec3 gEyeWorldPos;                                                                  
uniform MaterialInfo material;

float CalcShadowFactor(vec4 LightSpacePos)                                                  
{                                                                                           
    vec3 ProjCoords = LightSpacePos.xyz / LightSpacePos.w;                                  
    vec2 UVCoords;                                                                          
    UVCoords.x = 0.5 * ProjCoords.x + 0.5;                                                  
    UVCoords.y = 0.5 * ProjCoords.y + 0.5;                                                  
    float Depth = Color0.a;                                          
    if (Depth <= (ProjCoords.z + 0.005))                                                    
        return 0.5;                                                                         
    else                                                                                    
        return 1.0;                                                                         
} 

vec4 CalcLightInternal(vec3 LightDirection, VSOutput In, float ShadowFactor)            
{                                                                                           
    float DiffuseFactor = dot(In.Normal, -LightDirection);
	vec3 VertexToEye = normalize(gEyeWorldPos - In.WorldPos);                             
    vec3 LightReflect = normalize(reflect(LightDirection, In.Normal));                     
    float SpecularFactor = dot(VertexToEye, LightReflect);                              
    SpecularFactor = pow(SpecularFactor, material.shininess);   

		
	vec3 CoolColour = vec3(0.31, 0.29, 0.5);
	vec3 WarmColour = vec3(0.5, 0.14, 0.14);
	vec3 cool = min(CoolColour, 1.0);
	vec3 warm = min(WarmColour, 1.0);
    
	vec3 colour = min(mix(cool, warm, DiffuseFactor), 1.0);

    float OutlineFactor = dot(VertexToEye, In.Normal);
	if (OutlineFactor < 0.2) colour = vec3(0,0,0);
		                           
    vec4 FinalColor = vec4(colour, 1.0);                                                                          
                                                                                           
                                                                                            
    return FinalColor;
}                                                                                           
                                                                                            
vec4 CalcDirectionalLight(VSOutput In)                                                      
{                                                                                           
    return CalcLightInternal(gDirectionalLight.Direction, In, 1.0);  
}                                                                                           
                                                                                            
vec4 CalcPointLight(PointLight l, VSOutput In)                                       
{                                                                                           
    vec3 LightDirection = In.WorldPos - l.Position;                                           
    float Distance = length(LightDirection);                                                
    LightDirection = normalize(LightDirection);                                             
    
	float ShadowFactor = 1;//CalcShadowFactor(In.LightSpacePos);
	                                                                               
    vec4 Color = CalcLightInternal(LightDirection, In, 1);                         
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
    In.Color = Color0;
	In.Normal   = normalize(Normal0);
    In.WorldPos = WorldPos0;
	In.LightSpacePos = LightSpacePos0;

    vec4 TotalLight;                                         
                                                                                            
    for (int i = 0 ; i < gNumPointLights ; i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], In);                              
    }                                                                                       
                                                                                            
    for (int i = 0 ; i < gNumSpotLights ; i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], In);                                
    }                                                                                       
                                                                                            
    FragColor = TotalLight;     
}
