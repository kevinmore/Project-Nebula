#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Utility/EngineCommon.h>
#include "Texture.h"

class Material : protected QOpenGLFunctions_4_3_Core
{
public:
	Material(const QString& name);
	Material(const QString& name,
			 const QColor& ambientColor,
			 const QColor& diffuseColor,
			 const QColor& specularColor,
			 const QColor& emissiveColor,
		     float shininess,
		     float shininessStrength,
		     int twoSided,
		     int blendMode,
		     bool alphaBlending);

	virtual ~Material();

	void setName(const QString& name) { m_name = name; }
	QString name() const { return m_name; }

	bool isTranslucent() const { return m_alphaBlending; }

	void addTexture(TexturePtr tex) { m_textures << tex; }

	void bind();

	void init();

	QString m_name;

	QColor m_ambientColor;
	QColor m_diffuseColor;
	QColor m_specularColor;
	QColor m_emissiveColor;

	float m_shininess;
	float m_shininessStrength;
	float m_roughness;
	float m_fresnelReflectance;

	int  m_twoSided;
	int  m_blendMode;
	bool m_alphaBlending;

	enum BlendMode 
	{
		Default  = 0x0,
		Additive = 0x1
	};


	QVector<TexturePtr> m_textures;
};

typedef QSharedPointer<Material> MaterialPtr;
