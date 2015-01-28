#pragma once
#include <QObject>
#include <Utility/EngineCommon.h>
#include <Primitives/Material.h>

class MaterialManager : QObject
{
public:
	MaterialManager(QObject* parent = 0);
	~MaterialManager();

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
							bool alphaBlending,
							bool hasTexture);
	
	void deleteMaterial(MaterialPtr material);

	void clear();

private:
	QMap<QString, MaterialPtr> m_materials;
};

