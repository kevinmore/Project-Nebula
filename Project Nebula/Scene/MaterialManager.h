#pragma once
#include <QtCore/QMap>
#include <QtCore/QString>
#include <Primitives/Material.h>

class MaterialManager
{
public:
	MaterialManager(void);
	~MaterialManager(void);

	Material* getMaterial(const QString& name);
	void addMaterial(const QString& name, Material *Mat);

private:
	QMap<QString, Material*> m_materials;

};

