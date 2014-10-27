#pragma once
#include <QtCore/QMap>
#include <QtCore/QString>
#include <QtCore/QSharedPointer>
#include <Primitives/Mesh.h>

typedef QSharedPointer<Mesh> MeshPtr;

class MeshManager
{
public:
	MeshManager(void);
	~MeshManager(void);

	MeshPtr getMesh(const QString& name);
	MeshPtr addMesh(const QString& name, unsigned int numIndices, unsigned int baseVertex, unsigned int baseIndex);

private:
	QMap<QString, MeshPtr> m_meshes;

};

