#include <gl/glew.h>
#include <Primitives/Material.h>

Material::Material(const QString& name,
				   const vec4& ambientColor,
				   const vec4& diffuseColor,
				   const vec4& specularColor,
				   const vec4& emissiveColor,
				   float shininess,
				   float shininessStrength,
				   int twoSided,
				   int blendMode,
				   bool alphaBlending,
				   bool hasTexture)
	: m_name(name),
	m_ambientColor(ambientColor),
	m_diffuseColor(diffuseColor),
	m_specularColor(specularColor),
	m_emissiveColor(emissiveColor),
	m_shininess(shininess),
	m_shininessStrength(shininessStrength),
	m_twoSided(twoSided),
	m_blendMode(blendMode),
	m_alphaBlending(alphaBlending),
	m_hasTexture(hasTexture)
{
	init();
}

Material::~Material(void)
{
}

void Material::init()
{
	const GLchar* uniformNames[7] =
	{
		"MaterialInfo.Ka",
		"MaterialInfo.Kd",
		"MaterialInfo.Ks",
		"MaterialInfo.Ke",
		"MaterialInfo.shininess",
		"MaterialInfo.shininessStrength",
		"MaterialInfo.hasTexture"
	};

	m_buffer = QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	m_buffer.create();
	m_buffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_buffer.bind();
	

// 	GLint offsets[7];
// 
// 
// 	m_buffer.allocate(sizeof(offsets));
	m_buffer.release();
}

void Material::bind()
{
	m_buffer.bind();

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

void Material::fillBuffer(QVector<GLubyte>& buffer, GLint* offsets)
{
// 	(reinterpret_cast<float*>(buffer.data() + offsets[0]))[0] = m_ambientColor.x();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[0]))[1] = m_ambientColor.y();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[0]))[2] = m_ambientColor.z();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[0]))[3] = m_ambientColor.w();
// 
// 	(reinterpret_cast<float*>(buffer.data() + offsets[1]))[0] = m_diffuseColor.x();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[1]))[1] = m_diffuseColor.y();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[1]))[2] = m_diffuseColor.z();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[1]))[3] = m_diffuseColor.w();
// 
// 	(reinterpret_cast<float*>(buffer.data() + offsets[2]))[0] = m_specularColor.x();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[2]))[1] = m_specularColor.y();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[2]))[2] = m_specularColor.z();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[2]))[3] = m_specularColor.w();
// 
// 	(reinterpret_cast<float*>(buffer.data() + offsets[3]))[0] = m_emissiveColor.x();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[3]))[1] = m_emissiveColor.y();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[3]))[2] = m_emissiveColor.z();
// 	(reinterpret_cast<float*>(buffer.data() + offsets[3]))[3] = m_emissiveColor.w();
// 
// 	*(reinterpret_cast<float*>(buffer.data() + offsets[4])) = m_shininess;
// 	*(reinterpret_cast<float*>(buffer.data() + offsets[5])) = m_shininessStrength;
// 
// 	*(reinterpret_cast<bool*>(buffer.data() + offsets[6])) = m_hasTexture;
}