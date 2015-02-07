#pragma once

#include <Utility/EngineCommon.h>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShaderProgram>

#define INVALID_LOCATION 0xffffffff

typedef QSharedPointer<QOpenGLShaderProgram> QOpenGLShaderProgramPtr;


class Technique : public QObject, protected QOpenGLFunctions_4_3_Core
{
public:

    Technique(const QString& shaderFileName = "");


    virtual bool init();

	QString shaderFileName() const { return m_shaderFileName; }
	void setShaderFilePath(const QString& path);
	void applyShader(const QString &shaderName);
    virtual void enable();


	QOpenGLShaderProgramPtr getShaderProgram() const { return m_shaderProgram; }
	GLuint getVAO() const { return m_VAO; };
	void setVAO(GLuint vaoID);

protected:
	virtual bool compileShader() = 0;
	QString m_shaderFilePath;
	QString m_shaderFileName;

    GLint getUniformLocation(const char* pUniformName);
    
    GLint getProgramParam(GLint param);
    
	QOpenGLShaderProgramPtr m_shaderProgram;
	GLuint m_VAO;
};


