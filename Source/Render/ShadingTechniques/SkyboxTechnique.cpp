#include "SkyboxTechnique.h"


SkyboxTechnique::SkyboxTechnique()
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

void SkyboxTechnique::setMVPMatrix(const mat4& MVP)
{
	m_shaderProgram->setUniformValue("gWVP", MVP);
}


void SkyboxTechnique::setTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gCubemapTexture", TextureUnit);
}

bool SkyboxTechnique::compileShader()
{
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + "skybox.vert");
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + "skybox.frag");
	m_shaderProgram->link();

	return true;
}
