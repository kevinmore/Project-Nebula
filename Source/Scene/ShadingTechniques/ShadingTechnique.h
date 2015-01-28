#pragma once

#include "Technique.h"
#include <Scene/Light.h>
#include <Primitives/Material.h>

class Scene;
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

    ShadingTechnique(Scene* scene, const QString &shaderName, ShaderType shaderType = STATIC);
	~ShadingTechnique() {}
    virtual bool init();

    void setMVPMatrix(const mat4& WVP);
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
	void setVertexColor(const QColor& col);
	void setMaterial(const Material& mat);

private:
    
	virtual bool compileShader();
	void initLights();

	Scene* m_scene;
	ShaderType m_shaderType;

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

typedef QSharedPointer<ShadingTechnique> ShadingTechniquePtr;