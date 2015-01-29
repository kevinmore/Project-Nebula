#include <Primitives/Material.h>

Material::Material( const QString& name )
	: m_name(name),
	m_ambientColor(Qt::black),
	m_diffuseColor(Qt::white),
	m_specularColor(Qt::white),
	m_emissiveColor(Qt::black),
	m_shininess(8),
	m_shininessStrength(1.0f),
	m_twoSided(1),
	m_blendMode(Default),
	m_alphaBlending(false)
{
	init();
}


Material::Material(const QString& name,
					const QColor& ambientColor,
					const QColor& diffuseColor,
					const QColor& specularColor,
					const QColor& emissiveColor,
					float shininess,
					float shininessStrength,
					int twoSided,
					int blendMode,
					bool alphaBlending)
	: m_name(name),
	  m_ambientColor(ambientColor),
	  m_diffuseColor(diffuseColor),
	  m_specularColor(specularColor),
	  m_emissiveColor(emissiveColor),
	  m_shininess(shininess),
	  m_shininessStrength(shininessStrength),
	  m_twoSided(twoSided),
	  m_blendMode(blendMode),
	  m_alphaBlending(alphaBlending)
{
	init();
}



Material::~Material() {}

void Material::init()
{
	Q_ASSERT(initializeOpenGLFunctions());
}

void Material::bind()
{
	// Specifies whether meshes using this material
	// must be rendered with or without back face culling
	(m_twoSided != 1) ? glDisable(GL_CULL_FACE) : glEnable(GL_CULL_FACE);

	if(m_alphaBlending && m_blendMode != -1)
	{
		switch(m_blendMode)
		{
		case Additive:
			glBlendFunc(GL_ONE, GL_ONE);
			break;

		case Default:
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			break;
		}
	}
}
