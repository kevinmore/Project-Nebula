#pragma once
#include <QObject>
#include <Utility/EngineCommon.h>
#include <Primitives/Texture.h>

class TextureManager : QObject
{
public:
	static TextureManager* instance();

	TexturePtr getTexture(const QString& name);
	TexturePtr addTexture(const QString& name, const QString& fileName, 
						  Texture::TextureType type = Texture::Texture2D, 
						  Texture::TextureUsage usage = Texture::DiffuseMap);
	void deleteTexture(const QString& name);
	void deleteTexture(TexturePtr texture);
	void clear();

//private:
	QMap<QString, TexturePtr> m_textures;

private:
	TextureManager(QObject* parent = 0);
	~TextureManager();
	static TextureManager* m_instance;
};

