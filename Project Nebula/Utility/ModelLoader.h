#pragma once
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtCore/QSharedPointer>
#include <Scene/AbstractModel.h>
#include <Utility/DataTypes.h>

class ModelLoader
{
public:
	ModelLoader();
	virtual ~ModelLoader();

	QVector<ModelDataPtr> loadModel(const QString& filename, const QOpenGLShaderProgramPtr& shaderProgram);
	QOpenGLVertexArrayObjectPtr getVAO();


private:
	MeshData loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const QString& filename, const aiMaterial* material);

	void prepareVertexBuffers();
	void prepareVertexContainers(unsigned int index, const aiMesh* mesh);

	QOpenGLBuffer m_vertexPositionBuffer;
	QOpenGLBuffer m_vertexColorBuffer;
	QOpenGLBuffer m_vertexTexCoordBuffer;
	QOpenGLBuffer m_vertexNormalBuffer;
	QOpenGLBuffer m_vertexTangentBuffer;
	QOpenGLBuffer m_indexBuffer;

	QVector<QVector3D> m_positions;
	QVector<QVector4D> m_colors;
	QVector<QVector2D> m_texCoords;
	QVector<QVector3D> m_normals;
	QVector<QVector3D> m_tangents;
	QVector<unsigned int> m_indices;

	QOpenGLVertexArrayObjectPtr m_vao;
	QOpenGLShaderProgramPtr     m_shaderProgram;

	/************************************************************************/
	/* Skinning Stuff                                                       */
	/************************************************************************/
public:
	uint NumBones() const
	{
		return m_NumBones;
	}

	void BoneTransform(float TimeInSeconds, QVector<mat4>& Transforms);

private:
#define NUM_BONES_PER_VEREX 4

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

		void AddBoneData(uint BoneID, float Weight);
	};

	void CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);    
	uint FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
	const aiNodeAnim* FindNodeAnim(const aiAnimation* pAnimation, const QString NodeName);
	void ReadNodeHeirarchy(float AnimationTime, const aiNode* pNode, const mat4& ParentTransform);
	void LoadBones(uint MeshIndex, const aiMesh* paiMesh);


	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	QVector<VertexBoneData> m_Bones;
	uint m_NumBones;
	QVector<BoneInfo> m_BoneInfo;
	mat4 m_GlobalInverseTransform;

	const aiScene* m_scene;
	QOpenGLBuffer m_vertexBoneBuffer;
};

