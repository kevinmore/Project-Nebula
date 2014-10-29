#pragma once

#include <list>
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLShaderProgram>
class Technique
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
    
	QOpenGLFunctions_4_3_Core* m_funcs;
	QOpenGLShaderProgram* m_shader;

private:

    typedef std::list<GLuint> ShaderObjList;
    ShaderObjList m_shaderObjList;
};


