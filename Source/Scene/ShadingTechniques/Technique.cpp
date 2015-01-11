#include <stdio.h>
#include <string.h>

#include "Technique.h"
#include <Utility/EngineCommon.h>

Technique::Technique( const QString& shaderFileName /*= ""*/ )
{
	m_shader = nullptr;
	m_shaderFilePath = "../Resource/Shaders/";
	m_shaderFileName = shaderFileName;
}


Technique::~Technique()
{
    m_shader->release();
	SAFE_DELETE(m_shader);
}


bool Technique::Init()
{
	m_shader = new QOpenGLShaderProgram();

	Q_ASSERT(initializeOpenGLFunctions());	

    return true;
}


void Technique::Enable()
{
    m_shader->bind();
}


void Technique::Disable()
{
	m_shader->release();
}

GLint Technique::GetUniformLocation(const char* pUniformName)
{
    GLuint location = m_shader->uniformLocation(pUniformName);

	if (location == INVALID_UNIFORM_LOCATION) 
	{
		qWarning() << "Warning! Unable to get the location of uniform" << pUniformName << "of Shader:" << m_shaderFileName;
	}

	return location;
}

GLint Technique::GetProgramParam(GLint param)
{
    GLint ret;
    glGetProgramiv(m_shader->programId(), param, &ret);
    return ret;
}

void Technique::SetShaderFilePath( const QString& path )
{
	m_shaderFilePath = path;
}

void Technique::ApplyShader( const QString &shaderName )
{
	m_shaderFileName = shaderName;
	compileShader();
}
