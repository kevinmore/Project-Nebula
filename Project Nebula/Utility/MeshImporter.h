#pragma once
#include <Primitives/MeshData.h>
#include <Primitives/Bone.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define INVALID_MATERIAL 0xFFFFFFFF

class MeshImporter
{
private:

	struct MeshEntry {
		MeshEntry()
		{
			NumIndices    = 0;
			BaseVertex    = 0;
			BaseIndex     = 0;
			MaterialIndex = INVALID_MATERIAL;
		}

		unsigned int NumIndices;
		unsigned int BaseVertex;
		unsigned int BaseIndex;
		unsigned int MaterialIndex;
	};


	QVector<MaterialInfo*> m_Materials;
	Bone* m_Root;
	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	uint m_NumBones;
	QVector<VertexBoneData> m_Bones;
	QVector<MeshEntry> m_Entries;
	QVector<BoneInfo> m_BoneInfo;
	mat4 m_GlobalInverseTransform;
	const aiAnimation* m_Animation;

public:
	const aiScene* m_pScene;
	QVector<MeshData*> m_Meshes;
	MeshImporter(void);
	~MeshImporter(void);
	void cleanUp();
	Bone* getSkeleton() { return m_Root; }

	void BoneTransform(float TimeInSeconds, QVector<mat4> &Transforms);

	const aiScene* loadMeshFromFile(const QString &fileName);
	bool processScene(const aiScene* pScene, const QString &Filename);
	MeshData* processMesh(uint MeshIndex, const aiMesh* paiMesh);
	void processBones(uint MeshIndex, const aiMesh *paiMesh, QVector<VertexBoneData> &Bones);
	MaterialInfo* processMaterial(const aiMaterial *pMaterial, const QString &Filename);
	void processSkeleton(const aiScene *scene, aiNode *node, Bone *parentNode, Bone &newNode);
	void ReadNodeHeirarchy(float AnimationTime, const aiNode* pNode, const mat4 &ParentTransform);
	const aiNodeAnim* FindNodeAnim(const aiAnimation* pAnimation, QString &NodeName);

	uint FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);    
};

