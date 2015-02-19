#pragma once

#include "Technique.h"
#include <Scene/Light.h>
#include <Primitives/Material.h>

class Scene;
class ShadingTechnique : public Technique 
{
	Q_OBJECT

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

	void setLight(Light::LightType type, uint index, const Light* light);

    void setMVPMatrix(const mat4& mvp);
    void setModelMatrix(const mat4& model);
	void setViewMatrix(const mat4& view);
    void setColorTextureUnit(uint textureUnit);
	void setShadowMapTextureUnit(uint textureUnit);
	void setNormalMapTextureUnit(uint textureUnit);
    void setCameraPosition(const vec3& cameraPos);
    
    void setBoneTransform(uint tndex, const mat4& transform);

	void setMaterial(const Material* mat);
	void setMatAmbientColor(const QColor& col);
	void setMatDiffuseColor(const QColor& col);
	void setMatSpecularColor(const QColor& col);
	void setMatEmissiveColor(const QColor& col);
	void setMatSpecularIntensity(float intensity);
	void setMatSpecularPower(float power);
	void setMatRoughnessValue(float val);
	void setMatFresnelReflectance(float val);
	void setMatRefractiveIndex(float val);

public slots:
	void updateLights();

private:
    
	virtual bool compileShader();

	Scene* m_scene;
	ShaderType m_shaderType;


	bool usingCubeMap;
};

typedef QSharedPointer<ShadingTechnique> ShadingTechniquePtr;