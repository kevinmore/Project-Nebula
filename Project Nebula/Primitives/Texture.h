#pragma once
#include <gl/glew.h>
#include <QtCore/QDebug>
#include <QtOpenGL/QGLWidget>
#include <QtCore/QString>
#include <Magick++.h>
#include <QtGui/QImage>


class Texture
{

public:
	enum TextureType
	{
		Texture1D      = GL_TEXTURE_1D,
		Texture2D      = GL_TEXTURE_2D,
		Texture3D      = GL_TEXTURE_3D,
		TextureCubeMap = GL_TEXTURE_CUBE_MAP
	};
	Texture(TextureType type = Texture2D);
	Texture(const QString& fileName, TextureType type = Texture2D);
	Texture(const QImage& image, TextureType type = Texture2D);
	virtual ~Texture();

	void bind(GLenum textureUnit);
	void release();
	void destroy();

	TextureType type() const { return m_type; }
	GLuint textureId() const { return m_textureId; }

private:
	bool load();

	QImage m_qimage;

	Magick::Image m_image;
	Magick::Blob  m_blob;

	QString     m_fileName;
	TextureType m_type;
	GLuint      m_textureId;
};

