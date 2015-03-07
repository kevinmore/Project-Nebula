#pragma once
#include <QObject>
#include <Utility/EngineCommon.h>
#include <Primitives/Material.h>

class MaterialManager : QObject
{
public:
	static MaterialManager* instance();

	MaterialPtr getMaterial(const QString& name);

	MaterialPtr addMaterial(const QString& name, 
							const QColor& ambientColor,
							const QColor& diffuseColor,
							const QColor& specularColor,
							const QColor& emissiveColor,
							float shininess,
							float shininessStrength,
							int twoSided,
							int blendMode,
							bool alphaBlending);
	
	void deleteMaterial(MaterialPtr material);

	void clear();

private:
	MaterialManager(QObject* parent = 0);
	~MaterialManager();
	static MaterialManager* m_instance;

	QMap<QString, MaterialPtr> m_materials;
};

