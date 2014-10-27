#pragma once
#include <Utility/DataTypes.h>
#include <QtGui/QOpenGLBuffer>
class Material
{
public:
	Material(const QString& name,
			 const vec4& ambientColor,
			 const vec4& diffuseColor,
			 const vec4& specularColor,
			 const vec4& emissiveColor,
		     float shininess,
		     float shininessStrength,
		     int twoSided,
		     int blendMode,
		     bool alphaBlending,
		     bool hasTexture,
			 GLuint programHandle);

	virtual ~Material(void);

	void setName(const QString& name) { m_name = name; }
	QString name() const { return m_name; }

	bool isTranslucent() const { return m_alphaBlending; }

	void bind();

private:
	void init();
	void fillBuffer(QVector<GLubyte>& buffer, GLint* offsets);

	QString m_name;

	vec4 m_ambientColor;
	vec4 m_diffuseColor;
	vec4 m_specularColor;
	vec4 m_emissiveColor;

	float m_shininess;
	float m_shininessStrength;

	int  m_twoSided;
	int  m_blendMode;
	bool m_alphaBlending;
	bool m_hasTexture;

	enum BlendMode 
	{
		Default  = 0x0,
		Additive = 0x1
	};

	QOpenGLBuffer m_buffer;
	GLuint m_programHandle;
};

