#pragma once

#include <list>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShaderProgram>
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
	GLuint getVAO() { return m_VAO; };
	void setVAO(GLuint vaoID) { m_VAO = vaoID; }

protected:
	virtual bool compileShader() = 0;
	QString m_shaderFilePath;
	QString m_shaderFileName;

    GLint GetUniformLocation(const char* pUniformName);
    
    GLint GetProgramParam(GLint param);
    
	QOpenGLShaderProgram* m_shader;
	GLuint m_VAO;
};


