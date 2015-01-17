#pragma once

#include "Technique.h"

class ShadingTechnique : public Technique 
{
public:

    static const uint MAX_POINT_LIGHTS = 2;
    static const uint MAX_SPOT_LIGHTS = 2;
    static const uint MAX_BONES = 200;

	enum ShaderType
	{
		STATIC,
		RIGGED
	};

    ShadingTechnique(const QString &shaderName, ShaderType shaderType = STATIC);
	~ShadingTechnique() {}
    virtual bool init();

    void setWVP(const mat4& WVP);
	void setLightWVP(const mat4& LightWVP);
    void setWorldMatrix(const mat4& WVP);
    void setColorTextureUnit(uint TextureUnit);
	void setShadowMapTextureUnit(uint TextureUnit);
	void setNormalMapTextureUnit(uint TextureUnit);
    void setDirectionalLight(const DirectionalLight& Light);
    void setPointLights(uint NumLights, const PointLight* pLights);
    void setSpotLights(uint NumLights, const SpotLight* pLights);
    void setEyeWorldPos(const vec3& EyeWorldPos);
    void setMatSpecularIntensity(float Intensity);
    void setMatSpecularPower(float Power);
    void setBoneTransform(uint Index, const mat4& Transform);

private:
    
	virtual bool compileShader();

	ShaderType m_shaderType;

    GLuint m_WVPLocation;
	GLuint m_LightWVPLocation;
    GLuint m_WorldMatrixLocation;
    GLuint m_colorTextureLocation;
	GLuint m_shadowMapLocation;
	GLuint m_normalMapLocation;
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


