#pragma once
#include <GL/glew.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <QtCore/QVector>
#include <QtGui/QOpenGLBuffer>
#include <QtGui/QOpenGLVertexArrayObject>
#include <QtCore/QSharedPointer>
#include <Scene/AbstractModel.h>

class ModelLoader
{
public:
	ModelLoader(void);
	~ModelLoader(void);

	QVector<ModelDataPtr> loadModel(const QString& filename, const QOpenGLShaderProgramPtr& shaderProgram);
	QOpenGLVertexArrayObjectPtr getVAO();


private:
	MeshData loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh);
	MaterialData loadMaterial(unsigned int index, const aiMaterial* material);
	TextureData  loadTexture(const QString& filename, const aiMaterial* material);

	void prepareVertexBuffers();
	void prepareVertexContainers(const aiMesh* mesh);

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
};

