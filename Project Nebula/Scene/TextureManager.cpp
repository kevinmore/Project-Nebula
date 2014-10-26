#include <Scene/TextureManager.h>


TextureManager::TextureManager(void)
{
}


TextureManager::~TextureManager(void)
{
}

Texture* TextureManager::getTexture( const QString& name )
{
	if(m_textures.find(name) != m_textures.end()) return m_textures[name];
	else return NULL;
}

void TextureManager::addTexture( const QString& name, Texture *tex )
{
	// if texture is already in the map
	if (m_textures.find(name) != m_textures.end())
	{
		m_textures[name] = tex;
	} 
	else return;
}
