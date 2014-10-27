#pragma once
#include <GL/glew.h>
#include <Primitives/MeshData.h>
#include <Primitives/Bone.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#define INVALID_MATERIAL 0xFFFFFFFF


struct BoneInfo
{
	mat4 boneOffset;
	mat4 finalTransformation;        

	BoneInfo()
	{ 
		boneOffset.fill(0);
		finalTransformation.fill(0);
	}
};



class MeshImporter
{
private:

	enum VB_TYPES {
		INDEX_BUFFER,
		POS_VB,
		NORMAL_VB,
		TEXCOORD_VB,
		BONE_VB,
		NUM_VBs            
	};

	GLuint m_VAO;
	GLuint m_Buffers[NUM_VBs];

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
	
	QVector<MeshEntry> m_Entries;

	// vertex attribute
	QVector<Texture*> m_Textures;

	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	QVector<BoneInfo> m_BoneInfo;
	uint m_NumBones;
	mat4 m_GlobalInverseTransform;

	QVector<MaterialInfo*> m_Materials;
	QVector<aiAnimation*> m_Animations;
	const aiScene* m_pScene;
	Assimp::Importer importer;

	uint m_offSet;
	bool m_loaded;
	Bone* m_Root;
	MyMeshData* m_wholeMesh;
public:
	
	QVector<MyMeshData*> m_Meshes;
	MeshImporter(void);
	~MeshImporter(void);
	void Render();
	void cleanUp();
	MyMeshData* getWholeMesh() { return m_wholeMesh; }
	Bone* getSkeleton() { return m_Root; }
	void processSkeleton(const aiScene *scene, aiNode *node, Bone *parentNode, Bone &newNode);

	void BoneTransform(float TimeInSeconds, QVector<mat4> &Transforms);
	
	bool loadSucceeded() { return m_loaded; };
	bool loadMeshFromFile(const QString &fileName);
	bool processScene(const aiScene* pScene, const QString &Filename);

	void processMesh(uint MeshIndex,
					const aiMesh* paiMesh,
					QVector<vec3>& Positions,
					QVector<vec3>& Normals,
					QVector<vec2>& TexCoords,
					QVector<VertexBoneData>& Bones,
					QVector<unsigned int>& Indices);

	void generateWholeMesh();

	void processBones(uint MeshIndex, const aiMesh *paiMesh, QVector<VertexBoneData> &Bones);
	MaterialInfo* processMaterial(const aiMaterial *pMaterial, const QString &Filename);
	aiAnimation* processAnimations(uint animationIndex);
	void ReadNodeHeirarchy(float AnimationTime, const aiNode* pNode, const mat4 &ParentTransform);
	const aiNodeAnim* FindNodeAnim(const aiAnimation* pAnimation, QString &NodeName);
	

	uint FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);    
};

