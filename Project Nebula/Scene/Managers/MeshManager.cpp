#include "MeshManager.h"


MeshManager::MeshManager(void)
{
}


MeshManager::~MeshManager(void)
{
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
	
	Mesh* mesh = new Mesh(name, numIndices, baseVertex, baseIndex);

	m_meshes[name] = MeshPtr(mesh);

	delete mesh;
	mesh = nullptr;

	return m_meshes[name];
}
