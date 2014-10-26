#include <Scene/MaterialManager.h>


MaterialManager::MaterialManager(void)
{
}


MaterialManager::~MaterialManager(void)
{
}

Material* MaterialManager::getMaterial( const QString& name )
{
	if(m_materials.find(name) != m_materials.end()) return m_materials[name];
	else return NULL;
}

void MaterialManager::addMaterial( const QString& name, Material *mat )
{
	// if material is already in the map
	if (m_materials.find(name) != m_materials.end())
	{
		m_materials[name] = mat;
	} 
	else return;
}
