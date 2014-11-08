#pragma once

#include "technique.h"
#include <Utility/DataTypes.h>

struct BaseLight
{
    vec3 Color;
    float AmbientIntensity;
    float DiffuseIntensity;

    BaseLight()
    {
        Color = vec3(0.0f, 0.0f, 0.0f);
        AmbientIntensity = 0.0f;
        DiffuseIntensity = 0.0f;
    }
};

struct DirectionalLight : public BaseLight
{        
    vec3 Direction;

    DirectionalLight()
    {
        Direction = vec3(0.0f, 0.0f, 0.0f);
    }
};

struct PointLight : public BaseLight
{
    vec3 Position;

    struct
    {
        float Constant;
        float Linear;
        float Exp;
    } Attenuation;

    PointLight()
    {
        Position = vec3(0.0f, 0.0f, 0.0f);
        Attenuation.Constant = 1.0f;
        Attenuation.Linear = 0.0f;
        Attenuation.Exp = 0.0f;
    }
};

struct SpotLight : public PointLight
{
    vec3 Direction;
    float Cutoff;

    SpotLight()
    {
        Direction = vec3(0.0f, 0.0f, 0.0f);
        Cutoff = 0.0f;
    }
};

class ShadingTechnique : public Technique 
{
public:

    static const uint MAX_POINT_LIGHTS = 2;
    static const uint MAX_SPOT_LIGHTS = 2;
    static const uint MAX_BONES = 200;

    ShadingTechnique(const QString &vertShaderFile, const QString &fragShaderFile);

    virtual bool Init();

    void SetWVP(const mat4& WVP);
    void SetWorldMatrix(const mat4& WVP);
    void SetColorTextureUnit(uint TextureUnit);
    void SetDirectionalLight(const DirectionalLight& Light);
    void SetPointLights(uint NumLights, const PointLight* pLights);
    void SetSpotLights(uint NumLights, const SpotLight* pLights);
    void SetEyeWorldPos(const vec3& EyeWorldPos);
    void SetMatSpecularIntensity(float Intensity);
    void SetMatSpecularPower(float Power);
    void SetBoneTransform(uint Index, const mat4& Transform);

private:
    
	QString m_vertShaderFile, m_fragShaderFile;

    GLuint m_WVPLocation;
    GLuint m_WorldMatrixLocation;
    GLuint m_colorTextureLocation;
    GLuint m_eyeWorldPosLocation;
    GLuint m_matSpecularIntensityLocation;
    GLuint m_matSpecularPowerLocation;
    GLuint m_numPointLightsLocation;
    GLuint m_numSpotLightsLocation;

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Direction;
    } m_dirLightLocation;

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        struct {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } m_pointLightsLocation[MAX_POINT_LIGHTS];

    struct {
        GLuint Color;
        GLuint AmbientIntensity;
        GLuint DiffuseIntensity;
        GLuint Position;
        GLuint Direction;
        GLuint Cutoff;
        struct {
            GLuint Constant;
            GLuint Linear;
            GLuint Exp;
        } Atten;
    } m_spotLightsLocation[MAX_SPOT_LIGHTS];
    
    GLuint m_boneLocation[MAX_BONES];
};


