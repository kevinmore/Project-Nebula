#version 430

const int MAX_POINT_LIGHTS = 2;
const int MAX_SPOT_LIGHTS = 2;

in vec4 LightSpacePos0; 
in vec2 TexCoord0;
in vec3 Normal0;                                                       
in vec3 WorldPos0;                                                           
in vec3 Tangent0;

out vec4 FragColor;

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
uniform sampler2D gNormalMap;                                                                
uniform vec3 gEyeWorldPos;                                                                  
uniform float gMatSpecularIntensity;                                                        
uniform float gSpecularPower; 

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
    vec4 AmbientColor = vec4(Light.Color, 1.0) * Light.AmbientIntensity;                   
    float DiffuseFactor = dot(In.Normal, -LightDirection);                                     
                                                                                            
    vec4 DiffuseColor  = vec4(0, 0, 0, 0);                                                  
    vec4 SpecularColor = vec4(0, 0, 0, 0);                                                  
                                                                                            
    if (DiffuseFactor > 0.0) {                                                                
        DiffuseColor = vec4(Light.Color, 1.0) * Light.DiffuseIntensity * DiffuseFactor;    
                                                                                            
        vec3 VertexToEye = normalize(gEyeWorldPos - In.WorldPos);                             
        vec3 LightReflect = normalize(reflect(LightDirection, In.Normal));                     
        float SpecularFactor = dot(VertexToEye, LightReflect);                              
        SpecularFactor = pow(SpecularFactor, gSpecularPower);                               
        if (SpecularFactor > 0.0) {                                                           
            SpecularColor = vec4(Light.Color, 1.0) *                                       
                            gMatSpecularIntensity * SpecularFactor;                         
        }                                                                                   
    }                                                                                       
                                                                                            
    return (AmbientColor + ShadowFactor * (DiffuseColor + SpecularColor)); 
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
    
	float ShadowFactor = CalcShadowFactor(In.LightSpacePos);
	                                                                               
    vec4 Color = CalcLightInternal(l.Base, LightDirection, In, ShadowFactor);                         
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
    In.TexCoord = TexCoord0;
    //In.Normal   = normalize(Normal0);
	In.Normal   = CalcBumpedNormal();
    In.WorldPos = WorldPos0;
	In.LightSpacePos = LightSpacePos0;

    vec4 TotalLight = CalcDirectionalLight(In);                                         
                                                                                            
    for (int i = 0 ; i < gNumPointLights ; i++) {                                           
        TotalLight += CalcPointLight(gPointLights[i], In);                              
    }                                                                                       
                                                                                            
    for (int i = 0 ; i < gNumSpotLights ; i++) {                                            
        TotalLight += CalcSpotLight(gSpotLights[i], In);                                
    }                                                                                       
                                                                                            
    FragColor = texture(gColorMap, In.TexCoord.xy) * TotalLight;     
}
