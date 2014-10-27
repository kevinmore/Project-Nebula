#include "TextureManager.h"


TextureManager::TextureManager(void)
{
}


TextureManager::~TextureManager(void)
{
}

TexturePtr TextureManager::getTexture( const QString& name )
{
	if(m_textures.find(name) != m_textures.end()) return m_textures[name];
	else return TexturePtr();
}

TexturePtr TextureManager::addTexture( const QString& name, const QString& fileName )
{
	// if texture is already in the map
	if (m_textures.find(name) != m_textures.end())
	{
		return m_textures[name];
	} 
	
	m_textures[name] = TexturePtr(new Texture(fileName));

	return m_textures[name];
}
