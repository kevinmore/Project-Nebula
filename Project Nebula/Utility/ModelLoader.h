#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtCore/QSharedPointer>
#include <Scene/AbstractModel.h>
#include <Utility/DataTypes.h>
#include <Animation/Rig/Skeleton.h>
#include <assert.h>
#include <QtGui/QOpenGLFunctions_4_3_Core>


#define POSITION_LOCATION    0
#define TEX_COORD_LOCATION   1
#define NORMAL_LOCATION      2
#define BONE_ID_LOCATION     3
#define BONE_WEIGHT_LOCATION 4


class ModelLoader : protected QOpenGLFunctions_4_3_Core
{
public:
	ModelLoader();
	virtual ~ModelLoader();

	/*
	 *	This is the core functionality
	 */
	QVector<ModelDataPtr> loadModel(const QString& filename);


	/*
	 *	Public Getters
	 */
	QOpenGLVertexArrayObject* getVAO() { return m_vao; };
	uint getNumBones() const {	return m_NumBones;	}
	QMap<QString, uint> getBoneMap() const { return m_BoneMapping; }
	mat4 getGlobalInverseTransform () const { return m_GlobalInverseTransform; }
	aiAnimation** getAnimations() const	{ return m_scene->mAnimations; }
	QVector<BoneInfo> getBoneInfo() const {	return m_BoneInfo; }
	aiNode* getRootNode() const	{ return m_scene->mRootNode; }


private:
	/*
	 *	Methods to process the model file
	 */
	MeshData loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const QString& filename, const aiMaterial* material);
	void loadBones(uint MeshIndex, const aiMesh* paiMesh);
	void prepareVertexContainers(unsigned int index, const aiMesh* mesh);
	Skeleton* generateSkeleton(aiNode* pAiRootNode, Skeleton* pRootSkeleton);

	/*
	 *	Clean up
	 */
	void clear();


	/*
	 *	Vertex Data Containers
	 */
	QVector<QVector3D> m_positions;
	QVector<QVector4D> m_colors;
	QVector<QVector2D> m_texCoords;
	QVector<QVector3D> m_normals;
	QVector<QVector3D> m_tangents;
	QVector<unsigned int> m_indices;



	/*
	 *	Vertex Buffers
	 */

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
	QOpenGLVertexArrayObject* m_vao;
	void prepareVertexBuffers();

	/*
	 *	Members Variables
	 */
	Assimp::Importer m_importer;
	const aiScene* m_scene;
	Skeleton* m_skeleton;
	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	QVector<VertexBoneData> m_Bones;
	uint m_NumBones;
	QVector<BoneInfo> m_BoneInfo;
	mat4 m_GlobalInverseTransform;

};


