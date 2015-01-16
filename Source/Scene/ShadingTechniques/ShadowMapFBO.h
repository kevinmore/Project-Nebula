#pragma once
#include <QOpenGLFunctions_4_3_Core>

class ShadowMapFBO : protected QOpenGLFunctions_4_3_Core
{
public:
	ShadowMapFBO();
	~ShadowMapFBO();

	bool init(uint windowWidth, uint windowHeight);
	void bindForWriting();
	void bindForReading(GLenum textureUnit);

private:
	GLuint m_fbo;
	GLuint m_shadowMap;
};

