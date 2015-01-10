#pragma once
#include <QString>
#include <QSharedPointer>
#include <QVector4D>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <Scene/GameObject.h>

struct MeshData
{
	QString name;

	unsigned int numIndices;
	unsigned int baseVertex;
	unsigned int baseIndex;
};


struct TextureData
{
	QString colorMap, normalMap;
	bool hasTexture;
};

struct MaterialData
{
	QString name;

	QVector4D ambientColor;
	QVector4D diffuseColor;
	QVector4D specularColor;
	QVector4D emissiveColor;

	float shininess;
	float shininessStrength;

	int twoSided;
	int blendMode;
	bool alphaBlending;
};

struct ModelData
{
	MeshData     meshData;
	TextureData  textureData;
	MaterialData materialData;
	bool hasAnimation;
	float animationDuration;
};

typedef QSharedPointer<MeshData> MeshDataPtr;
typedef QSharedPointer<TextureData> TextureDataPtr;
typedef QSharedPointer<MaterialData> MaterialDataPtr;
typedef QSharedPointer<ModelData> ModelDataPtr;


typedef QSharedPointer<QOpenGLShaderProgram> QOpenGLShaderProgramPtr;
typedef QSharedPointer<QOpenGLVertexArrayObject> QOpenGLVertexArrayObjectPtr;

class AbstractModel
{


public:
	AbstractModel(const QString& fileName, GameObject* go);
	virtual ~AbstractModel() = 0;

	virtual void render(float time) = 0;

	const QString fileName() { return m_fileName; }
	GameObject* gameObject() { return m_actor; }

protected:
	QString m_fileName;
	GameObject* m_actor;
};