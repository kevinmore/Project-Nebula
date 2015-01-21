#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <QString>
#include <Magick++.h>
#include <QImage>
#include <QSharedPointer>

class CubemapTexture : protected QOpenGLFunctions_4_3_Core
{
public:
	CubemapTexture(const QString& PosXFilename,
				   const QString& NegXFilename,
				   const QString& PosYFilename,
				   const QString& NegYFilename,
				   const QString& PosZFilename,
				   const QString& NegZFilename);

	~CubemapTexture();

	void bind(GLenum textureUnit);

private:
	void init();
	bool load();
	void destroy();

	Magick::Image m_image;
	Magick::Blob  m_blob;

	QVector<QString> m_fileNames;
	GLuint m_textureId;
};

typedef QSharedPointer<CubemapTexture> CubemapTexturePtr;