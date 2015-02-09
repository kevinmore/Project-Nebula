#version 430

// Material properties
struct MaterialInfo
{
    vec4 Ke; // Emissive reflectivity
};

out vec4 FragColor;
uniform MaterialInfo material;

void main()
{
    FragColor = material.Ke;
}
