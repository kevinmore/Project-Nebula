#include "MaterialManager.h"


MaterialManager::MaterialManager(QObject* parent)
	: QObject(parent)
{
}


MaterialManager::~MaterialManager(){}

MaterialPtr MaterialManager::getMaterial( const QString& name )
{
	if(m_materials.find(name) != m_materials.end()) return m_materials[name];
	else return MaterialPtr();
}

MaterialPtr MaterialManager::addMaterial(const QString& name, 
										const QColor& ambientColor,
										const QColor& diffuseColor,
										const QColor& specularColor,
										const QColor& emissiveColor,
										float shininess,
										float shininessStrength,
										int twoSided,
										int blendMode,
										bool alphaBlending,
										bool hasTexture)
{
	// if material is already in the map
	if (m_materials.find(name) != m_materials.end())
	{
		return m_materials[name];
	} 
	
	m_materials[name] = MaterialPtr(new Material(name, ambientColor, diffuseColor, specularColor, emissiveColor, shininess, shininessStrength, twoSided, blendMode, alphaBlending, hasTexture));

	return m_materials[name];
}

void MaterialManager::clear()
{
	m_materials.clear();
}

void MaterialManager::deleteMaterial( MaterialPtr material )
{
	m_materials.take(m_materials.key(material));
}
