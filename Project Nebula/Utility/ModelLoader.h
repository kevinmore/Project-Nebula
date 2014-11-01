#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtCore/QSharedPointer>
#include <Scene/AbstractModel.h>
#include <Utility/DataTypes.h>
#include <assert.h>
#include <QtGui/QOpenGLFunctions_4_3_Core>

class ModelLoader : protected QOpenGLFunctions_4_3_Core
{
public:
	ModelLoader();
	virtual ~ModelLoader();

	QVector<ModelDataPtr> loadModel(const QString& filename);
	QOpenGLVertexArrayObject* getVAO();


private:
	MeshData loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const QString& filename, const aiMaterial* material);
	void loadBones(uint MeshIndex, const aiMesh* paiMesh);

	void prepareVertexBuffers();
	void prepareVertexContainers(unsigned int index, const aiMesh* mesh);

	void clear();

	Assimp::Importer m_importer;
	const aiScene* m_scene;

	QVector<QVector3D> m_positions;
	QVector<QVector4D> m_colors;
	QVector<QVector2D> m_texCoords;
	QVector<QVector3D> m_normals;
	QVector<QVector3D> m_tangents;
	QVector<unsigned int> m_indices;

	QOpenGLVertexArrayObject* m_vao;


#define NUM_BONES_PER_VEREX 4
#define POSITION_LOCATION    0
#define TEX_COORD_LOCATION   1
#define NORMAL_LOCATION      2
#define BONE_ID_LOCATION     3
#define BONE_WEIGHT_LOCATION 4

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

	/************************************************************************/
	/* Skinning Stuff                                                       */
	/************************************************************************/
public:

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

	uint getNumBones() const
	{
		return m_NumBones;
	}

	QMap<QString, uint> getBoneMap() const
	{
		return m_BoneMapping;
	}

	mat4 getGlobalInverseTransform () const
	{
		return m_GlobalInverseTransform;
	}

	aiAnimation** getAnimations() const
	{
		return m_scene->mAnimations;
	}

	QVector<BoneInfo> getBoneInfo() const
	{
		return m_BoneInfo;
	}

	aiNode* getRootNode() const
	{
		return m_scene->mRootNode;
	}

	struct VertexBoneData
	{        
		uint IDs[NUM_BONES_PER_VEREX];
		float Weights[NUM_BONES_PER_VEREX];

		VertexBoneData()
		{
			Reset();
		};

		void Reset()
		{
			ZERO_MEM(IDs);
			ZERO_MEM(Weights);        
		}

		void AddBoneData(uint BoneID, float Weight)
		{
			for (uint i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(IDs) ; i++) {
				if (Weights[i] == 0.0) {
					IDs[i]     = BoneID;
					Weights[i] = Weight;
					return;
				}        
			}

			// should never get here - more bones than we have space for
			assert(0);
		}
	};



	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	QVector<VertexBoneData> m_Bones;
	uint m_NumBones;
	QVector<BoneInfo> m_BoneInfo;
	mat4 m_GlobalInverseTransform;

};


