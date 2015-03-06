#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QString>
#include <QSharedPointer>

#define DIFFUSE_TEXTURE_UNIT GL_TEXTURE0
#define SHADOW_TEXTURE_UNIT  GL_TEXTURE1
#define NORMAL_TEXTURE_UNIT  GL_TEXTURE2

struct FIBITMAP;
class Texture : protected QOpenGLFunctions_4_3_Core
{
public:

	enum TextureType
	{
		Texture1D      = GL_TEXTURE_1D,
		Texture2D      = GL_TEXTURE_2D,
		Texture3D      = GL_TEXTURE_3D,
		TextureCubeMap = GL_TEXTURE_CUBE_MAP
	};

	enum TextureUsage
	{
		DiffuseMap,
		SpecularMap,
		AmbientMap,
		EmissiveMap,
		HeightMap,
		NormalMap,
		ShininessMap,
		OpacityMap,
		DisplacementMap,
		LightMap,
		ReflectionMap,
		ShadowMap
	};

	Texture(const QString& fileName, TextureType type = Texture2D, TextureUsage usage = DiffuseMap);
	virtual ~Texture();

	void bind(GLenum textureUnit);
	void release();

	TextureType type() const { return m_type; }
	TextureUsage usage() const { return m_usage; }
	GLuint textureId() const { return m_textureId; }
	QString fileName() const { return m_fileName; }

	QPixmap generateQPixmap();

private:
	void init();
	bool load();
	void destroy();
	QImage& QImageNone();

	//pointer to the image, once loaded
	FIBITMAP* m_image;

	QString m_fileName;
	TextureType m_type;
	TextureUsage m_usage;
	GLuint m_textureId;
};

typedef QSharedPointer<Texture> TexturePtr;
