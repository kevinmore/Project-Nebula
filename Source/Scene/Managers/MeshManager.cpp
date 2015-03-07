#include "MeshManager.h"

MeshManager::MeshManager(QObject* parent) 
	: QObject(parent)
{}

MeshManager::~MeshManager() {}

MeshManager* MeshManager::m_instance = 0;

MeshManager* MeshManager::instance()
{
	static QMutex mutex;
	if (!m_instance)
	{
		QMutexLocker locker(&mutex);
		if (!m_instance)
			m_instance = new MeshManager;
	}

	return m_instance;
}

MeshPtr MeshManager::getMesh( const QString& name )
{
	if(m_meshes.find(name) != m_meshes.end()) return m_meshes[name];
	else return MeshPtr();
}

MeshPtr MeshManager::addMesh( const QString& name, unsigned int numIndices, unsigned int baseVertex, unsigned int baseIndex )
{
	// if mesh is already in the map
	if (m_meshes.find(name) != m_meshes.end())
	{
		return m_meshes[name];
	} 
	
	m_meshes[name] = MeshPtr(new Mesh(name, numIndices, baseVertex, baseIndex));

	return m_meshes[name];
}

void MeshManager::clear()
{
	m_meshes.clear();
}

void MeshManager::deleteMesh( MeshPtr mesh )
{
	m_meshes.take(m_meshes.key(mesh));
}
