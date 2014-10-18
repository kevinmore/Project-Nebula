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

	bool loadMeshFromFile(const QString &fileName);
	void getBufferData(QVector<float> **vertices, QVector<float> **normals,
		QVector<unsigned int> **indices);

	Bone* getSkeleton() { return m_Root; }

	MaterialInfo* processMaterial(aiMaterial *material);
	MeshData* processMesh(aiMesh *mesh);
	void processSkeleton(const aiScene *scene, aiNode *node, Bone *parentNode, Bone &newNode);

	void transformToUnitCoordinates();
	void findObjectDimensions(Bone *node, QMatrix4x4 transformation, QVector3D &minDimension, QVector3D &maxDimension);

};

