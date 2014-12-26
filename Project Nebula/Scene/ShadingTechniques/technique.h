#pragma once

#include <list>
#include <QtGui/QOpenGLFunctions_4_3_Core>
#include <QtGui/QOpenGLShaderProgram>
#define INVALID_UNIFORM_LOCATION 0xffffffff

class Technique : protected QOpenGLFunctions_4_3_Core
{
public:

    Technique(const QString& shaderFileName = "");

    virtual ~Technique();

    virtual bool Init();

	void SetShaderFilePath(const QString& path);
	void ApplyShader(const QString &shaderName);
    void Enable();
	void Disable();


	QOpenGLShaderProgram* getShader() { return m_shader; }

protected:
	virtual bool compileShader() = 0;
	QString m_shaderFilePath;
	QString m_shaderFileName;

    GLint GetUniformLocation(const char* pUniformName);
    
    GLint GetProgramParam(GLint param);
    
	QOpenGLShaderProgram* m_shader;
};


