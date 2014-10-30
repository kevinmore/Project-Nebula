#pragma once

#include <list>
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLShaderProgram>
class Technique : protected QOpenGLFunctions_4_3_Core
{
public:

    Technique();

    virtual ~Technique();

    virtual bool Init();

    void Enable();

	QOpenGLShaderProgram* getShader() { return m_shader; }

protected:

    bool Finalize();

    GLint GetUniformLocation(const char* pUniformName);
    
    GLint GetProgramParam(GLint param);
    
	QOpenGLShaderProgram* m_shader;
};


