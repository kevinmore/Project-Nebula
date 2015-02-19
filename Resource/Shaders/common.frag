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
    vec4 EmissiveColor = material.Ke;                                                                                                                                      
    vec4 DiffuseColor;                                                  
    vec4 SpecularColor;          
	                                        
    vec3 eyeDir  = normalize(gEyeWorldPos - In.WorldPos);     

	/////////////////////////// Calculate the Diffuse Color //////////////////////////////
	// calculate intermediary values                        
    float NdotL = dot(In.Normal, -LightDirection);
	float NdotV = dot(In.Normal, eyeDir);

	float angleVN = acos(NdotV);
    float angleLN = acos(NdotL);

	float alpha = max(angleVN, angleLN);
    float beta = min(angleVN, angleLN);
	float gamma = dot(eyeDir - In.Normal *NdotV, LightDirection - In.Normal * NdotL);
       
	float roughnessSquared = material.roughnessValue  * material.roughnessValue ;     
	float roughnessSquared9 = (roughnessSquared / (roughnessSquared + 0.09));

	// calculate C1, C2 and C3
    float C1 = 1.0 - 0.5 * (roughnessSquared / (roughnessSquared + 0.33));
    float C2 = 0.45 * roughnessSquared9;

	if(gamma >= 0.0)
    {
        C2 *= sin(alpha);
    }
    else
    {
        C2 *= (sin(alpha) - pow((2.0 * beta) / PI, 3.0));
    }

	float powValue = (4.0 * alpha * beta) / (PI * PI);
    float C3  = 0.125 * roughnessSquared9 * powValue * powValue;

	// now calculate both main parts of the formula
    float A = gamma * C2 * tan(beta);
    float B = (1.0 - abs(gamma)) * C3 * tan((alpha + beta) / 2.0);

	// put it all together
    float L1 = max(0.0, NdotL) * (C1 + A + B);

	// also calculate interreflection
    float twoBetaPi = 2.0 * beta / PI;

	//TODO: p is squared in this case... how to separate this?
    float L2 = 0.17 * max(0.0, NdotL) * (roughnessSquared / (roughnessSquared + 0.13)) * (1.0 - gamma * twoBetaPi * twoBetaPi);

	DiffuseColor = material.Kd * Light.Color * Light.Intensity * (L1 + L2);	


	/////////////////////////// Calculate the Specular Color //////////////////////////////
	float specularFactor = 0.0;
    if (NdotL > 0.0) 
	{   
        // calculate intermediary values
        vec3 halfVector = normalize( eyeDir - LightDirection);
        float NdotH = max(dot(In.Normal, halfVector), 0.0); 
        float NdotV = max(dot(In.Normal, eyeDir), 0.0); // note: this could also be NdotL, which is the same value
        float VdotH = max(dot(eyeDir, halfVector), 0.0);
		
		// geometric attenuation
        float NH2 = 2.0 * NdotH;
        float g1 = (NH2 * NdotV) / VdotH;
        float g2 = (NH2 * NdotL) / VdotH;
        float geoAtt = min(1.0, min(g1, g2));
		
		 // roughness (or: microfacet distribution function)
        // beckmann distribution function
        float r1 = 1.0 / ( 4.0 * roughnessSquared * pow(NdotH, 4.0));
        float r2 = (NdotH * NdotH - 1.0) / (roughnessSquared * NdotH * NdotH);
        float roughness = r1 * exp(r2);
		
		// fresnel
        // Schlick approximation
        float fresnel = pow(1.0 - VdotH, 5.0);
        fresnel *= (1.0 - material.fresnelReflectance);
        fresnel += material.fresnelReflectance;              
		
		specularFactor = (fresnel * geoAtt * roughness) / (NdotV * NdotL * 3.14);                                                               
    }
	specularFactor = max(specularFactor, 0.01);                                                                                       
	SpecularColor = material.Ks * Light.Color * NdotL * specularFactor * material.shininessStrength;
    return (AmbientColor + EmissiveColor + ShadowFactor * (DiffuseColor + SpecularColor)); 
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
	In.Color = Color0;
    In.TexCoord = TexCoord0;
    In.WorldPos = WorldPos0;
	In.Tangent = Tangent0;

	if(material.hasNormalMap)
		In.Normal   = CalcBumpedNormal();
	else
		In.Normal   = normalize(Normal0);

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
