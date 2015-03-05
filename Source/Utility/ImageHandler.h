#pragma once
#include <QImageIOHandler>
#include <QImage>
#include <QPixmap>
#include <QDebug>
#include <FreeImage.h>

class ImageHandler : public QImageIOHandler
{
public:
	ImageHandler();
	~ImageHandler();

	bool canRead() const;
	bool read(QImage *image);

	QByteArray name() const;

	QVariant option(ImageOption option) const;

	void setOption(ImageOption option, const QVariant &value);

	bool supportsOption(ImageOption option) const;

public:
	static FreeImageIO& fiio();
	static FREE_IMAGE_FORMAT getFIF(QIODevice *device, const QByteArray& fmt);    

	static QImage& QImageNone();
	static bool isQImageNone(const QImage& qi);
	static QVector<QRgb>& PaletteNone();
	static bool isPaletteNone(const QVector<QRgb> &pal);
	static QImage FIBitmapToQImage(FIBITMAP *dib);
	static QPixmap FIBitmapToQPixmap(FIBITMAP *dib);
	static QVector<QRgb> getPalette(FIBITMAP *dib);
};

