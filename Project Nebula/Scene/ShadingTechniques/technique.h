#pragma once

#include <list>
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLShaderProgram>
#define INVALID_UNIFORM_LOCATION 0xffffffff

class Technique : protected QOpenGLFunctions_4_3_Core
{
public:

    Technique();

    virtual ~Technique();

    virtual bool Init();

    void Enable();
	void Disable();


	QOpenGLShaderProgram* getShader() { return m_shader; }

protected:

    bool Finalize();

    GLint GetUniformLocation(const char* pUniformName);
    
    GLint GetProgramParam(GLint param);
    
	QOpenGLShaderProgram* m_shader;
};


