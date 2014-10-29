#include <stdio.h>
#include <string.h>

#include "technique.h"

Technique::Technique()
{
	m_shader = nullptr;
	m_funcs = nullptr;
}


Technique::~Technique()
{
    m_shader->release();
}


bool Technique::Init()
{
	m_shader = new QOpenGLShaderProgram();

	QOpenGLContext* context = QOpenGLContext::currentContext();

	Q_ASSERT(context);

	m_funcs = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
	m_funcs->initializeOpenGLFunctions();

    return true;
}


void Technique::Enable()
{
    m_shader->bind();
}


GLint Technique::GetUniformLocation(const char* pUniformName)
{
    return m_shader->uniformLocation(pUniformName);
}

GLint Technique::GetProgramParam(GLint param)
{
    GLint ret;
    m_funcs->glGetProgramiv(m_shader->programId(), param, &ret);
    return ret;
}
