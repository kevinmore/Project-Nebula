#include <Primitives/Texture.h>
#include <QDebug>
#include <QGLWidget>
#include <assert.h>

Texture::Texture(const QString& fileName, TextureType type, TextureUsage usage)
	: m_qimage(),
	  m_image(),
	  m_fileName(fileName),
	  m_type(type),
	  m_usage(usage),
	  m_textureId(0)
{
	init();
	load();
}

Texture::Texture(const QImage& image, TextureType type, TextureUsage usage)
	: m_qimage(QGLWidget::convertToGLFormat(image)),
	m_image(),
	m_fileName(""),
	m_type(type),
	m_usage(usage),
	m_textureId(0)
{
	init();
	load();
}


Texture::~Texture()
{
	if(QOpenGLContext::currentContext() && m_textureId != 0)
 		glDeleteTextures(1, &m_textureId);
}

void Texture::init()
{
	assert(initializeOpenGLFunctions());
}

bool Texture::load()
{
	glGenTextures(1, &m_textureId);
	glBindTexture(m_type, m_textureId);

	// use image magick
	if(!m_fileName.isEmpty())
	{
		try
		{
			m_image.read(m_fileName.toStdString());
			m_image.magick("RGBA");
			m_image.write(&m_blob);
		}
		catch (Magick::Error& e)
		{
			qDebug() << e.what();
			destroy();
			return false;
		}

		glTexImage2D(m_type,
			0,
			GL_RGBA,
			static_cast<GLsizei>(m_image.columns()),
			static_cast<GLsizei>(m_image.rows()),
			0,
			GL_RGBA,
			GL_UNSIGNED_BYTE,
			m_blob.data()
			);
	}
	// use qt image
	else
	{
		glTexImage2D(m_type,
			0,
			GL_RGBA,
			m_qimage.width(),
			m_qimage.height(),
			0,
			GL_RGBA,
			GL_UNSIGNED_BYTE,
			m_qimage.bits()
			);
	}

	glTexParameterf(m_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(m_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(m_type, 0);

	//qDebug() << "Loaded texture:" << m_fileName;

	return true;
}

void Texture::destroy()
{
	if(m_textureId)
	{
		glDeleteTextures(1, &m_textureId);
		m_textureId = 0;
	}
}

void Texture::bind(GLenum textureUnit)
{
	glActiveTexture(textureUnit);
	glBindTexture(m_type, m_textureId);
}

void Texture::release()
{
	glBindTexture(m_type, 0);
}

QPixmap Texture::generateQPixmap()
{
	QImage im(static_cast<const uchar *>(m_blob.data()), m_image.columns(), m_image.rows(), QImage::Format_ARGB32);

	QPixmap pix;
	pix.convertFromImage(im);

	return pix;
}
