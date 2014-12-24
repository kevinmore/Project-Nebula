#pragma once
#include <QMap>
#include <QString>
#include <QSharedPointer>
#include <Primitives/Texture.h>

typedef QSharedPointer<Texture> TexturePtr;

class TextureManager
{
public:
	TextureManager(void);
	~TextureManager(void);

	TexturePtr getTexture(const QString& name);
	TexturePtr addTexture(const QString& name, const QString& fileName, 
						  Texture::TextureType type = Texture::Texture2D, 
						  Texture::TextureUsage usage = Texture::ColorMap);

private:
	QMap<QString, TexturePtr> m_textures;

};

