#pragma once
#include <QObject>
#include <Utility/EngineCommon.h>
#include <Primitives/Mesh.h>

typedef QSharedPointer<Mesh> MeshPtr;

class MeshManager : QObject
{
public:
	MeshManager(QObject* parent = 0);
	~MeshManager();

	MeshPtr getMesh(const QString& name);
	MeshPtr addMesh(const QString& name, unsigned int numIndices, unsigned int baseVertex, unsigned int baseIndex);

	void deleteMesh(MeshPtr mesh);

	void clear();

private:
	QMap<QString, MeshPtr> m_meshes;

};

