#include "CubeMapTexture.h"
#include <QDebug>
#include <QPixmap>
#include <assert.h>
#include <FreeImage.h>

static const GLenum types[6] = 
{  GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };

CubemapTexture::CubemapTexture(const QString& PosXFilename,
							   const QString& NegXFilename,
							   const QString& PosYFilename,
							   const QString& NegYFilename,
							   const QString& PosZFilename,
							   const QString& NegZFilename)
							   : m_textureId(0)
{
	m_fileNames << PosXFilename << NegXFilename << PosYFilename << NegYFilename
		        << PosZFilename << NegZFilename;
	init();
	load();
}

CubemapTexture::~CubemapTexture()
{
	destroy();
}

void CubemapTexture::init()
{
	Q_ASSERT(initializeOpenGLFunctions());
}

bool CubemapTexture::load()
{
	glGenTextures(1, &m_textureId);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureId);

	for (int i = 0 ; i < 6 ; ++i) 
	{
		// load image using freeimage
		//image format
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		//pointer to the image, once loaded
		FIBITMAP* image;
		//pointer to the image data
		BYTE* bits(0);
		//image width and height
		unsigned int width(0), height(0);

		//check the file signature and deduce its format
		fif = FreeImage_GetFileType(m_fileNames[i].toStdString().c_str(), 0);
		//if still unknown, try to guess the file format from the file extension
		if(fif == FIF_UNKNOWN) 
			fif = FreeImage_GetFIFFromFilename(m_fileNames[i].toStdString().c_str());
		//if still unkown, return failure
		if(fif == FIF_UNKNOWN)
			return false;

		//check that the plugin has reading capabilities and load the file
		if(FreeImage_FIFSupportsReading(fif))
			image = FreeImage_Load(fif, m_fileNames[i].toStdString().c_str());
		//if the image failed to load, return failure
		if(!image)
			return false;

		//retrieve the image data
		bits = FreeImage_GetBits(image);
		//get the image width and height
		width = FreeImage_GetWidth(image);
		height = FreeImage_GetHeight(image);
		//if this somehow one of these failed (they shouldn't), return failure
		if((bits == 0) || (width == 0) || (height == 0))
			return false;


		int BPP = FreeImage_GetBPP(image); 
		int format = BPP == 24 ? GL_BGR : BPP == 8 ? GL_LUMINANCE : 0; 
		int internalFormat = BPP == 24 ? GL_RGB : GL_DEPTH_COMPONENT; 

		glTexImage2D(types[i], 0, GL_RGB, width, height, 0, format, GL_UNSIGNED_BYTE, bits);

		FreeImage_Unload(image);
	}    

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);           

	return true;
}

void CubemapTexture::destroy()
{
	if(QOpenGLContext::currentContext() && m_textureId)
	{
		glDeleteTextures(1, &m_textureId);
		m_textureId = 0;
	}
}

void CubemapTexture::bind( GLenum textureUnit )
{
	glActiveTexture(textureUnit);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureId);
}
