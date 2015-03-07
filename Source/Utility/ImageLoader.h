#pragma once
#include <QImage>
#include <Magick++.h>
#include <QGLWidget>

class ImageLoader
{
public:

	static ImageLoader* instance();

	bool processWithImageMagick(const QString& filename);
	QImage getQImage() { return m_qimage; }
	QImage getQImageInOpenGLFormat() { return QGLWidget::convertToGLFormat(m_qimage); }
	Magick::Image getImage() { return m_image; }
	Magick::Blob getBlob() { return m_blob; }

private:
	ImageLoader();
	~ImageLoader();
	static ImageLoader* m_instance;

	QImage m_qimage;
	Magick::Image m_image;
	Magick::Blob  m_blob;
};

