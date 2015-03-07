#include "ImageLoader.h"
#include <QDebug>

ImageLoader::ImageLoader()
{}


ImageLoader::~ImageLoader()
{}

ImageLoader* ImageLoader::m_instance = 0;

ImageLoader* ImageLoader::instance()
{
	static QMutex mutex;
	if (!m_instance) {
		QMutexLocker locker(&mutex);
		if (!m_instance)
			m_instance = new ImageLoader;
	}

	return m_instance;
}

bool ImageLoader::processWithImageMagick( const QString& filename )
{
	bool sucess = true;
	Magick::Image image;
	Magick::Blob  blob;

	try
	{
		image.read(filename.toStdString());
		image.magick("RGBA");
		image.write(&blob);
		m_image = image;
		m_blob = blob;

		return true;
	}
	catch (Magick::Error& e)
	{
		qWarning() << e.what();
		sucess = false;
	}

	if(!sucess)
	{
		m_qimage = QImage(filename);
	}

	return sucess;
}
