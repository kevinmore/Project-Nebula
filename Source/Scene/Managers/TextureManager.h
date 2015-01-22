#pragma once
#include <Utility/EngineCommon.h>
#include <Primitives/Texture.h>

class TextureManager
{
public:
	TextureManager(void);
	~TextureManager(void);

	TexturePtr getTexture(const QString& name);
	TexturePtr addTexture(const QString& name, const QString& fileName, 
						  Texture::TextureType type = Texture::Texture2D, 
						  Texture::TextureUsage usage = Texture::ColorMap);
	void deleteTexture(const QString& name);
	void deleteTexture(TexturePtr texture);
	void clear();

private:
	QMap<QString, TexturePtr> m_textures;
};

