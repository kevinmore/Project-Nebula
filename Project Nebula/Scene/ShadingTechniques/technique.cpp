#include <stdio.h>
#include <string.h>

#include "technique.h"

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
    return m_shader->uniformLocation(pUniformName);
}

GLint Technique::GetProgramParam(GLint param)
{
    GLint ret;
    glGetProgramiv(m_shader->programId(), param, &ret);
    return ret;
}
