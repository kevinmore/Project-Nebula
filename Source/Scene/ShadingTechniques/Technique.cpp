#include "Technique.h"
#include <Utility/EngineCommon.h>

Technique::Technique( const QString& shaderFileName /*= ""*/ )
	: m_shaderProgram(0),
	  m_shaderFilePath("../Resource/Shaders/"),
	  m_shaderFileName(shaderFileName)
{}


Technique::~Technique()
{
    m_shaderProgram->release();
}


bool Technique::init()
{
	m_shaderProgram = QOpenGLShaderProgramPtr(new QOpenGLShaderProgram);

	Q_ASSERT(initializeOpenGLFunctions());	

    return true;
}


void Technique::enable()
{
    m_shaderProgram->bind();
}


void Technique::disable()
{
	m_shaderProgram->release();
}

GLint Technique::getUniformLocation(const char* pUniformName)
{
    GLuint location = m_shaderProgram->uniformLocation(pUniformName);

	if (location == INVALID_UNIFORM_LOCATION) 
	{
		qWarning() << "Warning! Unable to get the location of uniform" << pUniformName << "of Shader:" << m_shaderFileName;
	}

	return location;
}

GLint Technique::getProgramParam(GLint param)
{
    GLint ret;
    glGetProgramiv(m_shaderProgram->programId(), param, &ret);
    return ret;
}

void Technique::setShaderFilePath( const QString& path )
{
	m_shaderFilePath = path;
}

void Technique::applyShader( const QString &shaderName )
{
	m_shaderFileName = shaderName;
	compileShader();
}

void Technique::setVAO( GLuint vaoID )
{
	m_VAO = vaoID;
}
