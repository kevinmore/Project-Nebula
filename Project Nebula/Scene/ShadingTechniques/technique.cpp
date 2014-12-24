#include <stdio.h>
#include <string.h>

#include "Technique.h"

Technique::Technique()
{
	m_shader = nullptr;
}


Technique::~Technique()
{
    m_shader->release();
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
		fprintf(stderr, "Warning! Unable to get the location of uniform '%s'\n", pUniformName);
	}

	return location;
}

GLint Technique::GetProgramParam(GLint param)
{
    GLint ret;
    glGetProgramiv(m_shader->programId(), param, &ret);
    return ret;
}
