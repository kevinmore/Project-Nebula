#version 430

const float PI = 3.14159;
const int MAX_DIRENCTIONAL_LIGHTS = 2;
const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

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
uniform vec3 gEyeWorldPos;                                                                  

vec4 CalcLightInternal(vec3 LightDirection, VSOutput In)            
{                                                                                           
    float DiffuseFactor = dot(In.Normal, -LightDirection);
	vec3 VertexToEye = normalize(gEyeWorldPos - In.WorldPos);                             
		
	vec3 CoolColour = vec3(0.62, 0.58, 1.0);
	vec3 WarmColour = vec3(1.0, 0.28, 0.28);
	vec3 cool = min(CoolColour, 1.0);
	vec3 warm = min(WarmColour, 1.0);
    
	vec3 colour = min(mix(cool, warm, DiffuseFactor), 1.0);

    float OutlineFactor = dot(VertexToEye, In.Normal);
	if (OutlineFactor < 0.2) colour = vec3(0,0,0);
		                           
    vec4 FinalColor = vec4(colour, 1.0);                                                                          
                                                                                           
                                                                                            
    return FinalColor;
}                                                                                           
                                                                                            
vec4 CalcDirectionalLight(DirectionalLight l, VSOutput In)                                                      
{                                                                                           
    return CalcLightInternal(l.Direction, In);  
}                                                                                           
                                                                                            
vec4 CalcPointLight(PointLight l, VSOutput In)                                       
{                                                                                           
    vec3 LightDirection = In.WorldPos - l.Position;                                           
    float Distance = length(LightDirection);                                                
    LightDirection = normalize(LightDirection);                                             
    
	float ShadowFactor = 1;//CalcShadowFactor(In.LightSpacePos);
	                                                                               
    vec4 Color = CalcLightInternal(LightDirection, In);                         
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
	In.Normal   = normalize(Normal0);
    In.WorldPos = WorldPos0;
	In.Tangent = Tangent0;

	vec4 TotalLight;

	for (int i = 0 ; i < gNumDirectionalLights ; i++) {                                           
        TotalLight += CalcDirectionalLight(gDirectionalLights[i], In);                              
    } 

	if(gNumDirectionalLights > 1)
		TotalLight /= gNumDirectionalLights;

    for (int i = 0 ; i < gNumPointLights ; i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], In);                              
    }                                                                                       
    
	if(gNumPointLights > 1)
		TotalLight /= gNumPointLights;
	                                                                                        
    for (int i = 0 ; i < gNumSpotLights ; i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], In);                                
    }                                                                                       
    
	if(gNumSpotLights > 1)
		TotalLight /= gNumSpotLights;
	                                                                                       
    FragColor = TotalLight;     
}
