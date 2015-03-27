#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <Utility/EngineCommon.h>
#include <Animation/Rig/Skeleton.h>
#include <assert.h>
#include <QOpenGLFunctions_4_3_Core>
#include <QSharedPointer>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/ConvexHullCollider.h>
#include <Snow/Cuda/CUDAVector.h>

struct MeshData
{
	QString name;

	unsigned int numIndices;
	unsigned int baseVertex;
	unsigned int baseIndex;
};


struct TextureData
{
	QString diffuseMap, normalMap, opacityMap;
};

struct MaterialData
{
	QString name;

	QColor ambientColor;
	QColor diffuseColor;
	QColor specularColor;
	QColor emissiveColor;

	float shininess;
	float shininessStrength;

	int twoSided;
	int blendMode;
	bool alphaBlending;

	TextureData textureData;
};

struct ModelData
{
	MeshData     meshData;
	MaterialData materialData;
	bool hasAnimation;
	float animationDuration;
};

typedef QSharedPointer<MeshData> MeshDataPtr;
typedef QSharedPointer<TextureData> TextureDataPtr;
typedef QSharedPointer<MaterialData> MaterialDataPtr;
typedef QSharedPointer<ModelData> ModelDataPtr;

struct cudaGraphicsResource;

class Scene;
class ModelLoader : protected QOpenGLFunctions_4_3_Core
{
public:
	ModelLoader();
	virtual ~ModelLoader();

	enum MODEL_TYPE
	{
		STATIC_MODEL,
		RIGGED_MODEL,
		COLLIDER
	};

	/*
	 *	This is the core functionality
	 */
	QVector<ModelDataPtr> loadModel(const QString& filename, GLuint shaderProgramID = 0, const QString& loadingFlags = "Quality");


	/*
	 *	Public Getters And Setters
	 */
	GLuint getVAO() const { return m_VAO; };
	uint getNumBones() const {	return m_NumBones;	}
	QMap<QString, uint> getBoneMap() const { return m_BoneMapping; }
	mat4 getGlobalInverseTransform () const { return m_GlobalInverseTransform; }
	aiAnimation** getAnimations() const	{ return m_aiScene->mAnimations; }
	QVector<Bone> getBoneInfo() const {	return m_BoneInfo; }
	aiNode* getRootNode() const	{ return m_aiScene->mRootNode; }
	Skeleton* getSkeletom() const { return m_skeleton; }
	ShadingTechniquePtr getRenderingEffect() const { return m_effect; }
	MODEL_TYPE getModelType() const { return m_modelType; }
	void setModelType(MODEL_TYPE type) { m_modelType = type; } 
	SphereCollider* getBoundingSphere();
	BoxCollider* getBoundingBox();
	ConvexHullCollider* getConvexHullCollider();
	cudaGraphicsResource* getCudaVBO() { return m_cudaVBO; }
	uint getNumFaces() const { return m_faces.size() / 3; }
	uint getNumVertices() const { return m_positions.size(); }

private:
	/*
	 *	Methods to process the model file
	 */
	MeshData     loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const aiMaterial* material);
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
		ModelFeatures()
		{
			hasColorMap = false;
			hasNormalMap = false;
		}
	} m_modelFeatures;

	/*
	 *	Vertex Data Containers
	 */
	QVector<vec3> m_positions;
	QVector<vec4> m_colors;
	QVector<vec2> m_texCoords;
	QVector<vec3> m_normals;
	QVector<vec3> m_tangents;
	QVector<uint> m_indices;

	/*
	 *	Face Container (for convex shape)
	 */
	QVector<CUDAVec3> m_faces;


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

	// bounding box limits
	float m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ;

	// CUDA Stuff
	cudaGraphicsResource *m_cudaVBO;
};

typedef QSharedPointer<ModelLoader> ModelLoaderPtr;
