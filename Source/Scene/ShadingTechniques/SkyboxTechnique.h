#pragma once

#include "Technique.h"

class SkyboxTechnique : public Technique 
{
public:
	SkyboxTechnique();
	~SkyboxTechnique();
	virtual bool init();

	void setWVP(const mat4& WVP);
	void setTextureUnit(uint TextureUnit);

private:
	virtual bool compileShader();

	GLuint m_WVPLocation;
	GLuint m_textureLocation;
};
typedef QSharedPointer<SkyboxTechnique> SkyboxTechniquePtr;
