#version 430

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

out vec4 FragColor;
uniform MaterialInfo material;

void main()
{
    FragColor = material.Ke;
}
