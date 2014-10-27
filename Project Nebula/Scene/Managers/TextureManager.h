#pragma once
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QSharedPointer>
#include <Primitives/Texture.h>

typedef QSharedPointer<Texture> TexturePtr;

class TextureManager
{
public:
	TextureManager(void);
	~TextureManager(void);

	TexturePtr getTexture(const QString& name);
	TexturePtr addTexture(const QString& name, const QString& fileName);

private:
	QMap<QString, TexturePtr> m_textures;

};

