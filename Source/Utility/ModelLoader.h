#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <Scene/AbstractModel.h>
#include <Utility/EngineCommon.h>
#include <Animation/Rig/Skeleton.h>
#include <assert.h>
#include <QOpenGLFunctions_4_3_Core>
#include <QSharedPointer>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

class Scene;
class ModelLoader : protected QOpenGLFunctions_4_3_Core
{
public:
	ModelLoader(Scene* scene);
	virtual ~ModelLoader();

	enum MODEL_TYPE
	{
		STATIC_MODEL,
		RIGGED_MODEL
	};

	/*
	 *	This is the core functionality
	 */
	QVector<ModelDataPtr> loadModel(const QString& filename, GLuint shaderProgramID = 0, const QString& loadingFlags = "Fast");


	/*
	 *	Public Getters
	 */
	GLuint getVAO() { return m_VAO; };
	uint getNumBones() const {	return m_NumBones;	}
	QMap<QString, uint> getBoneMap() const { return m_BoneMapping; }
	mat4 getGlobalInverseTransform () const { return m_GlobalInverseTransform; }
	aiAnimation** getAnimations() const	{ return m_aiScene->mAnimations; }
	QVector<Bone> getBoneInfo() const {	return m_BoneInfo; }
	aiNode* getRootNode() const	{ return m_aiScene->mRootNode; }
	Skeleton* getSkeletom() const { return m_skeleton; }
	ShadingTechniquePtr getRenderingEffect() { return m_effect; }
	MODEL_TYPE getModelType() { return m_modelType; }

private:
	/*
	 *	Methods to process the model file
	 */
	MeshData loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const QString& filename, const aiMaterial* material);
	void loadBones(uint MeshIndex, const aiMesh* paiMesh);
	void prepareVertexContainers(unsigned int index, const aiMesh* mesh);
	void generateSkeleton(aiNode* pAiRootNode, Bone* pRootSkeleton, mat4& parentTransform);
	GLuint installShader();

	/*
	 *	Clean up
	 */
	void clear();

	/*
	 *	Model Features
	 */
	struct ModelFeatures
	{
		bool hasColorMap;
		bool hasNormalMap;
	} m_modelFeatures;

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
		TANGENT_VB,
		COLOR_VB,
		TEXCOORD_VB,
		BONE_VB,
		NUM_VBs            
	};

	GLuint m_shaderProgramID;
	GLuint m_VAO;
	QVector<GLuint> m_Buffers;
	void prepareVertexBuffers();

	/*
	 *	Members Variables
	 */
	QString m_fileName;
	Scene* m_scene;
	Assimp::Importer m_importer;
	const aiScene* m_aiScene;
	Skeleton* m_skeleton;
	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	QVector<VertexBoneData> m_Bones;
	uint m_NumBones;
	QVector<Bone> m_BoneInfo;
	mat4 m_GlobalInverseTransform;
	MODEL_TYPE m_modelType;
	ShadingTechniquePtr m_effect;
};

typedef QSharedPointer<ModelLoader> ModelLoaderPtr;
