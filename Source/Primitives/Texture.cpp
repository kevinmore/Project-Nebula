#include <Primitives/Texture.h>
#include <QDebug>
#include <QGLWidget>
#include <assert.h>
#include <FreeImage.h>

Texture::Texture(const QString& fileName, TextureType type, TextureUsage usage)
	: 
	  m_fileName(fileName),
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
	// load image using freeimage
	//image format
	FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
	//pointer to the image, once loaded
	FIBITMAP* m_image;
	//pointer to the image data
	BYTE* bits(0);
	//image width and height
	unsigned int width(0), height(0);

	//check the file signature and deduce its format
	fif = FreeImage_GetFileType(m_fileName.toStdString().c_str(), 0);
	//if still unknown, try to guess the file format from the file extension
	if(fif == FIF_UNKNOWN) 
		fif = FreeImage_GetFIFFromFilename(m_fileName.toStdString().c_str());
	//if still unkown, return failure
	if(fif == FIF_UNKNOWN)
		return false;

	//check that the plugin has reading capabilities and load the file
	if(FreeImage_FIFSupportsReading(fif))
		m_image = FreeImage_Load(fif, m_fileName.toStdString().c_str());
	//if the image failed to load, return failure
	if(!m_image)
		return false;

	//retrieve the image data
	bits = FreeImage_GetBits(m_image);
	//get the image width and height
	width = FreeImage_GetWidth(m_image);
	height = FreeImage_GetHeight(m_image);
	//if this somehow one of these failed (they shouldn't), return failure
	if((bits == 0) || (width == 0) || (height == 0))
		return false;

	glGenTextures(1, &m_textureId);
	glBindTexture(m_type, m_textureId);

	int BPP = FreeImage_GetBPP(m_image); 
	int format = BPP == 24 ? GL_BGR : BPP == 8 ? GL_LUMINANCE : 0; 
	int internalFormat = BPP == 24 ? GL_RGB : GL_DEPTH_COMPONENT; 

	glTexImage2D(m_type, 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, bits);

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

	//Free FreeImage's copy of the data
//	FreeImage_Unload(m_image);
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
	QImage im;//(static_cast<const uchar *>(m_blob.data()), m_image.columns(), m_image.rows(), QImage::Format_RGB32);

	QPixmap pix;
	pix.convertFromImage(im);

	return pix;
}
