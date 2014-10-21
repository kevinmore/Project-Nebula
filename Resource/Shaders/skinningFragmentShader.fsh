#version 330

uniform vec4 ambientColor;
uniform vec4 diffuseColor;
uniform vec4 specularColor;
uniform float ambientReflection;
uniform float diffuseReflection;
uniform float specularReflection;
uniform float shininess;
uniform bool useTexture;
uniform sampler2D texture;

in vec3 varyingNormal;
in vec3 varyingLightDirection;
in vec3 varyingViewerDirection;
in vec2 varyingTextureCoordinate;
in vec4 varyingColor;

out vec4 fragColor;

void main(void)
{
    vec3 normal = normalize(varyingNormal);
    vec3 lightDirection = normalize(varyingLightDirection);
    vec3 viewerDirection = normalize(varyingViewerDirection);
    vec4 ambientIllumination = ambientReflection * ambientColor;
    vec4 diffuseIllumination = diffuseReflection * max(0.0, dot(lightDirection, normal)) * diffuseColor;
    vec4 specularIllumination = specularReflection * pow(max(0.0, 
                                                             dot(-reflect(lightDirection, normal), viewerDirection)
                                                             ), shininess) * specularColor;
   
	if(useTexture)
	{
		fragColor = texture2D(texture, varyingTextureCoordinate) * (ambientIllumination + diffuseIllumination) + specularIllumination;
	}   
    else
	{
		fragColor = varyingColor * (ambientIllumination + diffuseIllumination) + specularIllumination;
	}
	
}