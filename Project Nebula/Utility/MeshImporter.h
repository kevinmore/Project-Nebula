#pragma once
#include <Primitives/MeshData.h>
#include <Primitives/Bone.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class MeshImporter
{
private:
	QVector<float> m_vertices;
	QVector<float> m_normals;
	QVector<unsigned int> m_indices;

	QVector<MaterialInfo*> m_materials;
	QVector<MeshData*> m_meshes;
	
	Bone* m_Root;
	
public:
	MeshImporter(void);
	~MeshImporter(void);
	void cleanUp();
	Bone* getSkeleton() { return m_Root; }
	bool loadMeshFromFile(const QString &fileName);
	MaterialInfo* processMaterial(aiMaterial *material, const QString &fileName);
	MeshData* processMesh(aiMesh *mesh);
	void processSkeleton(const aiScene *scene, aiNode *node, Bone *parentNode, Bone &newNode);

};

