#pragma once

#include "Technique.h"

class SkyboxTechnique : public Technique 
{
public:
	SkyboxTechnique();
	virtual bool init();

	void setMVPMatrix(const mat4& MVP);
	void setTextureUnit(uint TextureUnit);

private:
	virtual bool compileShader();

	GLuint m_WVPLocation;
	GLuint m_textureLocation;
};
typedef QSharedPointer<SkyboxTechnique> SkyboxTechniquePtr;
