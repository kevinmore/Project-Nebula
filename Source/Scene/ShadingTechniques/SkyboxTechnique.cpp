#include "SkyboxTechnique.h"


SkyboxTechnique::SkyboxTechnique()
{
}


SkyboxTechnique::~SkyboxTechnique()
{
}

bool SkyboxTechnique::init()
{
	if (!Technique::init()) 
	{
		return false;
	}

	return compileShader();
}

void SkyboxTechnique::setWVP(const mat4& WVP)
{
	glUniformMatrix4fv(m_WVPLocation, 1, GL_FALSE, WVP.data());    
}


void SkyboxTechnique::setTextureUnit(unsigned int TextureUnit)
{
	glUniform1i(m_textureLocation, TextureUnit);
}

bool SkyboxTechnique::compileShader()
{
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + "skybox.vert");
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + "skybox.frag");
	m_shaderProgram->link();

	m_WVPLocation = getUniformLocation("gWVP");
	m_textureLocation = getUniformLocation("gCubemapTexture");

	if (m_WVPLocation == INVALID_LOCATION ||
		m_textureLocation == INVALID_LOCATION) {
			return false;
	}

	return true;
}
