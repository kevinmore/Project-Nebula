#pragma once
#include <QtCore/QMap>
#include <QtCore/QString>
#include <Primitives/Texture.h>

class TextureManager
{
public:
	TextureManager(void);
	~TextureManager(void);

	Texture* getTexture(const QString& name);
	void addTexture(const QString& name, Texture *tex);

private:
	QMap<QString, Texture*> m_textures;

};

