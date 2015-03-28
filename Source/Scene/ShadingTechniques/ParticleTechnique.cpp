#include "ParticleTechnique.h"


ParticleTechnique::ParticleTechnique(const QString &shaderName, ShaderType shaderType)
	: Technique(shaderName),
	  m_shaderType(shaderType)
{
}

bool ParticleTechnique::init()
{
	if (!Technique::init() || m_shaderFileName.isEmpty()) 
	{
		return false;
	}

	return compileShader();
}


bool ParticleTechnique::compileShader()
{
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + m_shaderFileName + ".vert");
	
	switch(m_shaderType)
	{
	case UPDATE:
	{
		m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Geometry, m_shaderFilePath + m_shaderFileName + ".geom");
		const char* sVaryings[NUM_PARTICLE_ATTRIBUTES] = 
		{
			"vPositionOut",
			"vVelocityOut",
			"vColorOut",
			"fLifeTimeOut",
			"fSizeOut",
			"iTypeOut",
		};
		for (int i = 0; i < NUM_PARTICLE_ATTRIBUTES; ++i)
		{
			glTransformFeedbackVaryings(m_shaderProgram->programId(), 6, sVaryings, GL_INTERLEAVED_ATTRIBS);
		}
		break;
	}
		
	case RENDER:
		m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Geometry, m_shaderFilePath + m_shaderFileName + ".geom");
		m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + m_shaderFileName + ".frag");
		break;

	case SNOW:
		m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + m_shaderFileName + ".frag");
		break;
	}

	return m_shaderProgram->link();
}

