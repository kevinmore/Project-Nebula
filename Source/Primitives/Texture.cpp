#include <Primitives/Texture.h>
#include <QDebug>
#include <assert.h>
#include <Utility/ImageHandler.h>
#include <QGLWidget>

Texture::Texture(const QString& fileName, TextureType type, TextureUsage usage)
	: m_fileName(fileName),
	  m_type(type),
	  m_usage(usage),
	  m_textureId(0)
{
	init();
	load();
}

Texture::~Texture()
{
	destroy();
}

void Texture::init()
{
	Q_ASSERT(initializeOpenGLFunctions());
}

bool Texture::load()
{
	glGenTextures(1, &m_textureId);
	glBindTexture(m_type, m_textureId);

	// use qt first
	m_image = QImage(m_fileName);

	if (!m_image.isNull() && m_image.height() > 0 && m_image.width() > 0)
	{
		QImage gl_image = QGLWidget::convertToGLFormat(m_image);
		uint depth = gl_image.depth();
		uint channels = depth / 8;

		int format = channels == 4 ? GL_RGBA : channels == 3 ? GL_RGB : channels == 1 ? GL_LUMINANCE : 0; 
		int internalFormat = format; 
		if(format == GL_RGBA || format == GL_BGRA) internalFormat = GL_RGBA;
		if(format == GL_RGB || format == GL_BGR) internalFormat = GL_RGB;

		glTexImage2D(m_type, 0, internalFormat, gl_image.width(), gl_image.height(), 0, format, GL_UNSIGNED_BYTE, gl_image.bits());
	}
	else
	{
		// if failed to use qt, use freeimage
		std::string fileNameInStr = m_fileName.toStdString();
		const char* filename = fileNameInStr.c_str();

		// load image using freeimage
		//image format
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		FIBITMAP* bitmap(0);

		//image width and height
		unsigned int width(0), height(0);

		//check the file signature and deduce its format
		fif = FreeImage_GetFileType(filename, 0);
		//if still unknown, try to guess the file format from the file extension
		if(fif == FIF_UNKNOWN) 
			fif = FreeImage_GetFIFFromFilename(filename);
		//if still unkown, return failure
		if(fif == FIF_UNKNOWN)
			return false;

		//check that the plugin has reading capabilities and load the file
		if(FreeImage_FIFSupportsReading(fif))
			bitmap = FreeImage_Load(fif, filename);
		//if the image failed to load, return failure
		if(!bitmap)
			return false;

		//pointer to the image data
		//retrieve the image data
		BYTE* bits = FreeImage_GetBits(bitmap);
		//get the image width and height
		width = FreeImage_GetWidth(bitmap);
		height = FreeImage_GetHeight(bitmap);
		//if this somehow one of these failed (they shouldn't), return failure
		if((bits == 0) || (width == 0) || (height == 0))
			return false;

		uint depth = FreeImage_GetBPP(bitmap); 
		uint channels = depth / 8;

		int format = channels == 4 ? GL_RGBA : channels == 3 ? GL_RGB : channels == 1 ? GL_LUMINANCE : 0; 
		int internalFormat = format; 
		if(format == GL_RGBA || format == GL_BGRA) internalFormat = GL_RGBA;
		if(format == GL_RGB || format == GL_BGR) internalFormat = GL_RGB;

		glTexImage2D(m_type, 0, internalFormat, width, height, 0, format, GL_UNSIGNED_BYTE, bits);

		m_image = ImageHandler::FIBitmapToQImage(bitmap);

		//Free FreeImage's copy of the data
		FreeImage_Unload(bitmap);
	}

	glTexParameterf(m_type, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(m_type, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(m_type, 0);

	//qDebug() << "Loaded texture:" << m_fileName;

	return true;
}

void Texture::destroy()
{
	if(QOpenGLContext::currentContext() && m_textureId)
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
	QPixmap pix;
	pix.convertFromImage(m_image);
	return pix;
}
