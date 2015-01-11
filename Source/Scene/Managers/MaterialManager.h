#pragma once
#include <QMap>
#include <QString>
#include <QSharedPointer>
#include <Primitives/Material.h>

typedef QSharedPointer<Material> MaterialPtr;

class MaterialManager
{
public:
	MaterialManager(GLuint programHandle);
	~MaterialManager();

	MaterialPtr getMaterial(const QString& name);

	MaterialPtr addMaterial(const QString& name, 
							const QVector4D& ambientColor,
							const QVector4D& diffuseColor,
							const QVector4D& specularColor,
							const QVector4D& emissiveColor,
							float shininess,
							float shininessStrength,
							int twoSided,
							int blendMode,
							bool alphaBlending,
							bool hasTexture);

	void clear();

private:
	QMap<QString, MaterialPtr> m_materials;
	GLuint m_programHandle;
};

