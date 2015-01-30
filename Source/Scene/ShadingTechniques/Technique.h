#pragma once

#include <Utility/EngineCommon.h>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLShaderProgram>

#define INVALID_LOCATION 0xffffffff

typedef QSharedPointer<QOpenGLShaderProgram> QOpenGLShaderProgramPtr;
//typedef QSharedPointer<QOpenGLVertexArrayObject> QOpenGLVertexArrayObjectPtr;

struct BaseLight
{
	vec3 Color;
	float AmbientIntensity;
	float DiffuseIntensity;

	BaseLight()
	{
		Color = vec3(0.0f, 0.0f, 0.0f);
		AmbientIntensity = 0.0f;
		DiffuseIntensity = 0.0f;
	}
};

struct DirectionalLight : public BaseLight
{        
	vec3 Direction;

	DirectionalLight()
	{
		Direction = vec3(0.0f, 0.0f, 0.0f);
	}
};

struct PointLight : public BaseLight
{
	vec3 Position;

	struct
	{
		float Constant;
		float Linear;
		float Exp;
	} Attenuation;

	PointLight()
	{
		Position = vec3(0.0f, 0.0f, 0.0f);
		Attenuation.Constant = 1.0f;
		Attenuation.Linear = 0.0f;
		Attenuation.Exp = 0.0f;
	}
};

struct SpotLight : public PointLight
{
	vec3 Direction;
	float Cutoff;

	SpotLight()
	{
		Direction = vec3(0.0f, 0.0f, 0.0f);
		Cutoff = 0.0f;
	}
};

class Technique : protected QOpenGLFunctions_4_3_Core
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


