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

    ShadingTechnique(const QString &shaderName, ShaderType shaderType = STATIC, Scene* scene = 0);
    virtual bool init();
	virtual void enable();

	void setLight(uint index, const Light* light);

    void setMVPMatrix(const mat4& mvp);
    void setModelMatrix(const mat4& model);
	void setViewMatrix(const mat4& view);
    void setColorTextureUnit(uint TextureUnit);
	void setShadowMapTextureUnit(uint TextureUnit);
	void setNormalMapTextureUnit(uint TextureUnit);
    void setEyeWorldPos(const vec3& EyeWorldPos);
    
    void setBoneTransform(uint Index, const mat4& Transform);

	void setMaterial(const Material* mat);
	void setMatAmbientColor(const QColor& col);
	void setMatDiffuseColor(const QColor& col);
	void setMatSpecularColor(const QColor& col);
	void setMatEmissiveColor(const QColor& col);
	void setMatSpecularIntensity(float intensity);
	void setMatSpecularPower(float power);
	void setMatRoughnessValue(float val);
	void setMatFresnelReflectance(float val);

private:
    
	virtual bool compileShader();
	void initLights();

	Scene* m_scene;
	ShaderType m_shaderType;


	bool usingCubeMap;
};

typedef QSharedPointer<ShadingTechnique> ShadingTechniquePtr;