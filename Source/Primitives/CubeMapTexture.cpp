#include "CubeMapTexture.h"
#include <QDebug>
#include <QPixmap>
#include <assert.h>
#include <Utility/ImageLoader.h>
#include <QGLWidget>

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
							   :m_image(),
							   m_textureId(0)
{
	m_fileNames << PosXFilename << NegXFilename << PosYFilename << NegYFilename
		        << PosZFilename << NegZFilename;
	init();
	load();
}

CubemapTexture::~CubemapTexture()
{
	if(QOpenGLContext::currentContext() && m_textureId != 0)
		glDeleteTextures(1, &m_textureId);
}

void CubemapTexture::init()
{
	Q_ASSERT(initializeOpenGLFunctions());
}

bool CubemapTexture::load()
{
	glGenTextures(1, &m_textureId);
	glBindTexture(GL_TEXTURE_CUBE_MAP, m_textureId);
	ImageLoader* loader = ImageLoader::instance();

	for (int i = 0 ; i < 6 ; ++i) 
	{
		if (loader->processWithImageMagick(m_fileNames[i]))
		{
			m_image = loader->getImage();
			m_blob = loader->getBlob();

			QPixmap pix;
			pix.convertFromImage(loader->getQImage());
			m_qpixmaps << pix;

			glTexImage2D(types[i],
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
		else
		{
			QImage qimage = loader->getQImage();
			if (qimage.isNull() || qimage.width() == 0 || qimage.height() == 0)
			{
				destroy();
				return false;
			}
			qimage = QGLWidget::convertToGLFormat(qimage);

			glTexImage2D(types[i],
				0,
				GL_RGBA,
				qimage.width(),
				qimage.height(),
				0,
				GL_RGBA,
				GL_UNSIGNED_BYTE,
				qimage.bits()
				);
		}
		
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
	if(m_textureId)
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
